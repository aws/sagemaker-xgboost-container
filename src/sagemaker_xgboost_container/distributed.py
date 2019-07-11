# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import os
import socket
import time

from xgboost import rabit
import xgboost as xgb

# TODO: Point this to xgboost dmlc-core when merged upstream
from sagemaker_xgboost_container.dmlc_patch import tracker

logger = logging.getLogger('SageMakerRabit')
logging.basicConfig(level=logging.DEBUG)
LOCAL_HOSTNAME = '127.0.0.1'


class RabitHelper(object):
    def __init__(self, is_master, current_host, master_port):
        self.is_master = is_master
        self.rank = rabit.get_rank()
        self.current_host = current_host
        self.master_port = master_port

    def synchronize(self, data):
        # This function allows every node to share state with every other node
        # easily. This allows things like determining which nodes have data or not.
        results = []
        for i in range(rabit.get_world_size()):
            if self.rank == i:
                logger.debug("Broadcasting data from self ({}) to others".format(self.rank))
                rabit.broadcast(data, i)
                results.append(data)
            else:
                logger.debug("Receiving data from {}".format(i))
                message = rabit.broadcast(None, i)
                results.append(message)
        return results


class Rabit(object):
    def __init__(self,
                 hosts,
                 current_host=None,
                 master_host=None,
                 port=None,
                 max_connect_attempts=10,
                 connect_retry_timeout=3):
        """Set up rabit initialize.

        :param hosts: List of hostnames
        :param current_host: Current hostname. If not provided, use 127.0.0.1
        :param master_host: Master host hostname. If not provided, use alphabetically first hostname amongst hosts
                            to ensure all hosts use the same master
        :param port: Port to connect to master, if not specified use 9099
        :param max_connect_attempts:
        :param connect_retry_timeout:
        """
        # Get the host information. This is used to identify the master host
        # that will run the RabitTracker and also to work out how many clients/slaves
        # exist (this will ensure that all-reduce is set up correctly and that
        # it blocks whilst waiting for those hosts to process the data).
        self.hosts = sorted(hosts)
        self.n_workers = len(self.hosts)
        logger.debug("Found hosts: {} [{}]".format(self.hosts, self.n_workers))

        if not current_host:
            logger.debug("Setting current host as local.", current_host)
            current_host = LOCAL_HOSTNAME
        self.current_host = current_host

        # We use the first lexicographically named host as the master if not indicated otherwise
        if not master_host:
            master_host = self.hosts[0]
        self.master_host = master_host
        self.is_master_host = self.current_host == self.master_host

        logger.debug("Is Master: {}".format(self.is_master_host))
        logger.debug("Master: {}".format(self.master_host))

        # We start the RabitTracker on a known port on the first host. We can
        # do this since SageMaker Training instances are single tenent and we
        # don't need to worry about port contention.
        if port is None:
            port = 9099
            logger.debug("No port specified using: {}".format(port))
        else:
            logger.debug("Using provided port: {}".format(port))
        self.port = port

        self.max_connect_attempts = max_connect_attempts
        self.connect_retry_timeout = connect_retry_timeout

        self.stored_dmlc_env_vars = None

    def _write_dmlc_env_vars(self, env_vars, store_previous=False):
        previous_env_vars = {}
        for k, v in env_vars.items():
            if store_previous:
                if previous_env_vars.get(k):
                    logger.debug("Replacing env var {}={}".format(k, previous_env_vars[k]))
                    previous_env_vars[k] = os.environ[k]
                else:
                    previous_env_vars[k] = None
            if v:
                os.environ[k] = str(v)
            else:
                del os.environ[k]

        if store_previous:
            self.stored_dmlc_env_vars = previous_env_vars

    def start(self):
        """Start the rabit process.

        If current host is master host, initialize and start the Rabit Tracker in the background. All hosts then connect
        to the master host to set up Rabit rank.

        :return: Initialized RabitHelper, which includes helpful information such as is_master and port
        """
        self.rabit_context = None
        if self.is_master_host:
            # TODO: the Tracker script is very hacky and has a few bugs in it.
            # It really should be refactored and improved. Specifically it doesn't
            # make it easy to shutdown or reuse an existing tracker. This means
            # that two consecutive calls to Rabit() or the RabitTracker can
            # fail if the port is already in use.
            logger.debug("Master host. Starting Rabit Tracker.")

            # The Rabit Tracker is a Python script that is responsible for
            # allowing each instance of Rabit to find its peers and organize
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
            # These are the key-value config pairs that each of the Rabit slaves
            # should be initialized with. Since we have deterministically allocated
            # the master host, its port, and the number of workers, we don't need
            # to pass these out-of-band to each slave; but rely on the fact
            # that each slave will calculate the exact same config as the server.
            #
            # TODO: should probably check that these match up what we pass below.
            logger.info(self.rabit_context.slave_envs())

            # This actually starts the RabitTracker in a background/daemon thread
            # that will automatically exit when the main process has finished.
            self.rabit_context.start(self.n_workers)

        # Start each parameter server that connects to the master.
        logger.debug("Starting parameter server.")

        # Rabit runs as an in-process singleton library that can be configured once.
        # Calling this multiple times will cause a seg-fault (without calling finalize).
        # We pass it the environment variables that match up with the RabitTracker
        # so that this instance can discover its peers (and recover from failure).
        #
        # First we check that the RabitTracker is up and running. Rabit actually
        # breaks (at least on Mac OS X) if the server is not running before it
        # begins to try to connect (its internal retries fail because they reuse
        # the same socket instead of creating a new one).
        attempt = 0
        successful_connection = False
        while attempt < self.max_connect_attempts and not successful_connection:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    logger.debug("Checking if RabitTracker is available.")
                    s.connect((self.master_host, self.port))
                    successful_connection = True
                    logger.debug("Successfully connected to RabitTracker.")
                except OSError:
                    logger.info("Failed to connect to RabitTracker on attempt {}".format(attempt))
                    attempt += 1
                    logger.info("Sleeping for {} sec before retrying".format(self.connect_retry_timeout))
                    time.sleep(self.connect_retry_timeout)

        if not successful_connection:
            logger.error("Failed to connect to Rabit Tracker after %s attempts", self.max_connect_attempts)
            raise Exception("Failed to connect to Rabit Tracker")
        else:
            logger.info("Rabit Tracker is available for connections.")

        dmlc_env_vars = {
            'DMLC_NUM_WORKER': self.n_workers,
            'DMLC_TRACKER_URI': self.master_host,
            'DMLC_TRACKER_PORT': self.port,
            'DMLC_NUM_SERVER': '0'  # This starts RabitTracker instead of PSTracker in DMLC
        }

        self._write_dmlc_env_vars(dmlc_env_vars, store_previous=True)
        rabit.init()

        # We can check that the Rabit instance has successfully connected to the
        # server by getting the rank of the server (e.g. its position in the ring).
        # This should be unique for each instance.
        logger.debug("Rabit started - Rank {}".format(rabit.get_rank()))

        logger.debug("Executing user code")
        # We can now run user-code. Since XGBoost runs in the same process space
        # it will use the same instance of Rabit that we have configured. It has
        # a number of checks throughout the learning process to see if it is running
        # in distributed mode by calling Rabit APIs. If it is it will do the
        # synchronization automatically.
        #
        # Hence we can now execute any XGBoost specific training code and it
        # will be distributed automatically.
        return RabitHelper(self.is_master_host, self.current_host, self.port)

    def stop(self):
        """Shutdown parameter server.

        If current host is master host, also join the background thread that is running the master host.
        """
        logger.debug("Shutting down parameter server.")

        # This is the call that actually shuts down the Rabit server; and when
        # all of the slaves have been shut down then the RabitTracker will close
        # /shutdown itself.
        rabit.finalize()
        if self.is_master_host:
            self.rabit_context.join()

        if self.stored_dmlc_env_vars:
            self._write_dmlc_env_vars(self.stored_dmlc_env_vars)

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        return self.stop()


def train(train_cfg, dtrain, num_boost_round, evals, feval, early_stopping_rounds):
    bst = None
    current_boost_round = 0
    while current_boost_round < num_boost_round:
        bst = xgb.train(train_cfg, dtrain, num_boost_round=current_boost_round + 1, evals=evals,
                        feval=feval, early_stopping_rounds=early_stopping_rounds, xgb_model=bst)
    return bst
