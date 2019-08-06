# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""
Contains most of the wrapping code for XGBoost distributed training and Rabit support.
This is heavily inspired by the Dask version of XGBoost.
Some of this code should be made simpler once the XGBoost library is improved.
"""
import logging
from retrying import retry
import socket
import sys
import time

from xgboost import rabit

# This should point to xgb when the tracker is updated upstream
from sagemaker_xgboost_container.dmlc_patch import tracker

LOCAL_HOSTNAME = '127.0.0.1'


@retry(stop_max_delay=1000 * 60 * 15,
       wait_exponential_multiplier=100,
       wait_exponential_max=30000)
def _dns_lookup(host):
    """Retrying dns lookup on host """
    return socket.gethostbyname(host)


def wait_hostname_resolution(sm_hosts):
    """Wait for the hostname resolution of the container. This is known behavior as the cluster
    boots up and has been documented here:
     https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html#your-algorithms-training-algo-running-container-dist-training
    """
    for host in sm_hosts:
        _dns_lookup(host)


def rabit_run(exec_fun, args, include_in_training, hosts, current_host,
              first_port=None, second_port=None, max_connect_attempts=None,
              connect_retry_timeout=3, update_rabit_args=False):
    """Run execution function after initializing dmlc/rabit.

    This method initializes rabit twice:
        1. To broadcast to all hosts which hosts should be included in training.
        2. Run distributed xgb train() with just the hosts from above.

    :param exec_fun: Function to run while rabit is initialized. xgb.train() must run in the same process space
                    in order to utilize rabit initialization. Note that the execution function must also take the args
                    'is_distributed' and 'is_master'.
    :param args: Arguments to run execution function.
    :param include_in_training: Boolean if the current hosts should be used in training. This is done here so that
                                all the hosts in the cluster know which hosts to include during training.
    :param hosts:
    :param current_host:
    :param first_port: Port to use for the initial rabit initialization. If None, rabit defaults this to 9099
    :param second_port: Port to use for second rabit initialization. If None, this increments previous port by 1
    :param max_connect_attempts
    :param connect_retry_timeout
    :param update_rabit_args: Boolean to include rabit information to args. If True, the following is added:
                                is_master
    """
    with Rabit(
            hosts=hosts,
            current_host=current_host,
            port=first_port,
            max_connect_attempts=max_connect_attempts,
            connect_retry_timeout=connect_retry_timeout) as rabit:
        hosts_with_data = rabit.synchronize({'host': rabit.current_host, 'include_in_training': include_in_training})
        hosts_with_data = [record['host'] for record in hosts_with_data if record['include_in_training']]

        # Keep track of port used, so that hosts trying to shutdown know when server is not available
        previous_port = rabit.master_port

    if not include_in_training:
        logging.warning("Host {} not being used for distributed training.".format(current_host))
        sys.exit(0)

    second_rabit_port = second_port if second_port else previous_port + 1

    if len(hosts_with_data) > 1:
        # Set up rabit with nodes that have data and an unused port so that previous slaves don't confuse it
        # with the previous rabit configuration
        with Rabit(
                hosts=hosts_with_data,
                current_host=current_host,
                port=second_rabit_port,
                max_connect_attempts=max_connect_attempts,
                connect_retry_timeout=connect_retry_timeout) as cluster:
            if update_rabit_args:
                args.update({'is_master': cluster.is_master})
            exec_fun(**args)

    elif len(hosts_with_data) == 1:
        logging.debug("Only 1 host with training data, "
                      "starting single node training job from: {}".format(current_host))
        if update_rabit_args:
            args.update({'is_master': True})
        exec_fun(**args)

    else:
        raise RuntimeError("No hosts received training data.")


class RabitHelper(object):
    def __init__(self, is_master, current_host, master_port):
        """This is returned by the Rabit context manager for useful cluster information and data synchronization.

        :param is_master:
        :param current_host:
        :param master_port:
        """
        self.is_master = is_master
        self.rank = rabit.get_rank()
        self.current_host = current_host
        self.master_port = master_port

    def synchronize(self, data):
        """Synchronize data with the cluster.

        This function allows every node to share state with every other node easily.
        This allows things like determining which nodes have data or not.

        :param data: data to send to the cluster
        :return: aggregated data from the all the nodes in the cluster
        """
        results = []
        for i in range(rabit.get_world_size()):
            if self.rank == i:
                logging.debug("Broadcasting data from self ({}) to others".format(self.rank))
                rabit.broadcast(data, i)
                results.append(data)
            else:
                logging.debug("Receiving data from {}".format(i))
                message = rabit.broadcast(None, i)
                results.append(message)
        return results


class Rabit(object):

    @staticmethod
    def _get_logger(current_host):
        logging.basicConfig(format='%(name) [{}]: %(message)s'.format(current_host))
        return logging.getLogger('RabitContextManager')

    def __init__(self,
                 hosts,
                 current_host=None,
                 master_host=None,
                 port=None,
                 max_connect_attempts=None,
                 connect_retry_timeout=3):
        """Context manager for rabit initialization.

        :param hosts: List of hostnames
        :param current_host: Current hostname. If not provided, use 127.0.0.1.
        :param master_host: Master host hostname. If not provided, use alphabetically first hostname amongst hosts
                            to ensure determinism in choosing master node.
        :param port: Port to connect to master, if not specified use 9099.
        :param max_connect_attempts: Number of times to try connecting to RabitTracker. If this arg is set
                            to None, try indefinitely.
        :param connect_retry_timeout: Timeout value when attempting to connect to RabitTracker.
                            This will be ignored if max_connect_attempt is None
        """
        # Get the host information. This is used to identify the master host
        # that will run the RabitTracker and also to work out how many clients/slaves
        # exist (this will ensure that all-reduce is set up correctly and that
        # it blocks whilst waiting for those hosts to process the data).
        if not current_host:
            current_host = LOCAL_HOSTNAME
        self.current_host = current_host
        self.logger = self._get_logger(self.current_host)
        self.logger.debug("Found current host.")

        self.hosts = sorted(hosts)
        self.n_workers = len(self.hosts)
        self.logger.debug("Found hosts: {} [{}]".format(self.hosts, self.n_workers))

        # We use the first lexicographically named host as the master if not indicated otherwise
        if not master_host:
            master_host = self.hosts[0]
        self.master_host = master_host
        self.is_master_host = self.current_host == self.master_host

        self.logger.debug("Is Master: {}".format(self.is_master_host))
        self.logger.debug("Master: {}".format(self.master_host))

        # We start the RabitTracker on a known port on the first host. We can
        # do this since SageMaker Training instances are single tenent and we
        # don't need to worry about port contention.
        if port is None:
            port = 9099
            self.logger.debug("No port specified using: {}".format(port))
        else:
            self.logger.debug("Using provided port: {}".format(port))
        self.port = port

        if max_connect_attempts is None or max_connect_attempts > 0:
            self.max_connect_attempts = max_connect_attempts
        else:
            raise ValueError("max_connect_attempts must be None or an integer greater than 0.")
        self.connect_retry_timeout = connect_retry_timeout

    def start(self):
        """Start the rabit process.

        If current host is master host, initialize and start the Rabit Tracker in the background. All hosts then connect
        to the master host to set up Rabit rank.

        :return: Initialized RabitHelper, which includes helpful information such as is_master and port
        """
        self.rabit_context = None
        if self.is_master_host:
            self.logger.debug("Master host. Starting Rabit Tracker.")
            # The Rabit Tracker is a Python script that is responsible for
            # allowing each instance of rabit to find its peers and organize
            # itself in to a ring for all-reduce. It supports primitive failure
            # recovery modes.
            #
            # It runs on a master node that each of the individual Rabit instances
            # talk to.
            self.rabit_context = tracker.RabitTracker(hostIP=self.current_host,
                                                      nslave=self.n_workers,
                                                      port=self.port,
                                                      port_end=self.port + 1)

            # Useful logging to ensure that the tracker has started.
            # These are the key-value config pairs that each of the rabit slaves
            # should be initialized with. Since we have deterministically allocated
            # the master host, its port, and the number of workers, we don't need
            # to pass these out-of-band to each slave; but rely on the fact
            # that each slave will calculate the exact same config as the server.
            #
            # TODO: should probably check that these match up what we pass below.
            self.logger.info("Rabit slave environment: {}".format(self.rabit_context.slave_envs()))

            # This actually starts the RabitTracker in a background/daemon thread
            # that will automatically exit when the main process has finished.
            self.rabit_context.start(self.n_workers)

        # Start each parameter server that connects to the master.
        self.logger.debug("Starting parameter server.")

        # Rabit runs as an in-process singleton library that can be configured once.
        # Calling this multiple times will cause a seg-fault (without calling finalize).
        # We pass it the environment variables that match up with the RabitTracker
        # so that this instance can discover its peers (and recover from failure).
        #
        # First we check that the RabitTracker is up and running. Rabit actually
        # breaks (at least on Mac OS X) if the server is not running before it
        # begins to try to connect (its internal retries fail because they reuse
        # the same socket instead of creating a new one).
        #
        # if self.max_connect_attempts is None, this will loop indefinitely.
        attempt = 0
        successful_connection = False
        while (not successful_connection and
               (self.max_connect_attempts is None or attempt < self.max_connect_attempts)):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    self.logger.debug("Checking if RabitTracker is available.")
                    s.connect((self.master_host, self.port))
                    successful_connection = True
                    self.logger.debug("Successfully connected to RabitTracker.")
                except OSError:
                    self.logger.info("Failed to connect to RabitTracker on attempt {}".format(attempt))
                    attempt += 1
                    self.logger.info("Sleeping for {} sec before retrying".format(self.connect_retry_timeout))
                    time.sleep(self.connect_retry_timeout)

        if not successful_connection:
            self.logger.error("Failed to connect to Rabit Tracker after %s attempts", self.max_connect_attempts)
            raise Exception("Failed to connect to Rabit Tracker")
        else:
            self.logger.info("Connected to RabitTracker.")

        rabit.init(['DMLC_NUM_WORKER={}'.format(self.n_workers).encode(),
                    'DMLC_TRACKER_URI={}'.format(self.master_host).encode(),
                    'DMLC_TRACKER_PORT={}'.format(self.port).encode()])

        # We can check that the rabit instance has successfully connected to the
        # server by getting the rank of the server (e.g. its position in the ring).
        # This should be unique for each instance.
        self.logger.debug("Rabit started - Rank {}".format(rabit.get_rank()))
        self.logger.debug("Executing user code")

        # We can now run user-code. Since XGBoost runs in the same process space
        # it will use the same instance of rabit that we have configured. It has
        # a number of checks throughout the learning process to see if it is running
        # in distributed mode by calling rabit APIs. If it is it will do the
        # synchronization automatically.
        #
        # Hence we can now execute any XGBoost specific training code and it
        # will be distributed automatically.
        return RabitHelper(self.is_master_host, self.current_host, self.port)

    def stop(self):
        """Shutdown parameter server.

        If current host is master host, also join the background thread that is running the master host.
        """
        self.logger.debug("Shutting down parameter server.")

        # This is the call that actually shuts down the rabit server; and when
        # all of the slaves have been shut down then the RabitTracker will close
        # /shutdown itself.
        rabit.finalize()
        if self.is_master_host:
            self.rabit_context.join()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        return self.stop()
