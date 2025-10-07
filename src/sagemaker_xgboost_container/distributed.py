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
import socket
import sys
import json

from threading import Thread
from retrying import retry
from xgboost.tracker import RabitTracker
from xgboost import collective

LOCAL_HOSTNAME = "127.0.0.1"


@retry(stop_max_delay=1000 * 60 * 15, wait_exponential_multiplier=100, wait_exponential_max=30000)
def _dns_lookup(host):
    """Retrying dns lookup on host"""
    return socket.gethostbyname(host)


def wait_hostname_resolution(sm_hosts):
    """Wait for the hostname resolution of the container. This is known behavior as the cluster
    boots up and has been documented here:
     https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html#your-algorithms-training-algo-running-container-dist-training
    """
    for host in sm_hosts:
        _dns_lookup(host)


def rabit_run(
    exec_fun,
    args,
    include_in_training,
    hosts,
    current_host,
    first_port=None,
    second_port=None,
    max_connect_attempts=None,
    # TEST LN
    # connect_retry_timeout=10,
    connect_retry_timeout=3,
    update_rabit_args=False,
):
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
        connect_retry_timeout=connect_retry_timeout,
    ) as rabit_ctx:
        hosts_with_data = rabit_ctx.synchronize(
            {"host": rabit_ctx.current_host, "include_in_training": include_in_training}
        )
        hosts_with_data = [record["host"] for record in hosts_with_data if record["include_in_training"]]

        # Keep track of port used, so that hosts trying to shutdown know when server is not available
        previous_port = rabit_ctx.master_port

    if not include_in_training:
        logging.warning("Host {} not being used for distributed training.".format(current_host))
        sys.exit(0)

    second_rabit_port = second_port if second_port else previous_port + 1

    if len(hosts_with_data) > 1:
        # Set up rabit with nodes that have data and an unused port so that previous slaves don't confuse it
        # with the previous rabit configuration
        logging.info(f"SECOND_RABIT_DEBUG: hosts_with_data={hosts_with_data}, current_host={current_host}")

        with Rabit(
            hosts=hosts_with_data,
            current_host=current_host,
            port=second_rabit_port,
            max_connect_attempts=max_connect_attempts,
            connect_retry_timeout=connect_retry_timeout,
        ) as cluster:
            if update_rabit_args:
                logging.info(
                    f"RABIT_DEBUG: \
                             cluster.is_master={cluster.is_master}, \
                            current_host={current_host}"
                )

                args.update({"is_master": cluster.is_master})
            exec_fun(**args)

    elif len(hosts_with_data) == 1:
        logging.debug(
            "Only 1 host with training data, " "starting single node training job from: {}".format(current_host)
        )
        if update_rabit_args:
            args.update({"is_master": True})
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
        import time

        self.is_master = is_master  # Store hostname-based master determination
        self.current_host = current_host
        self.master_port = master_port
        self._id = int(time.time() * 1000000) % 1000000  # Unique ID for debugging
        logging.info(
            f"RABIT_HELPER_INIT: Created RabitHelper {self._id} with is_master={self.is_master} for host={current_host}"
        )

        try:
            self.rank = collective.get_rank()
            self.world_size = collective.get_world_size()
        except Exception:
            logging.error("collective init failed", exc_info=True)
            self.rank = 0
            self.world_size = 1

    def synchronize(self, data):
        """Synchronize data with the cluster.

        This function allows every node to share state with every other node easily.
        This allows things like determining which nodes have data or not.

        :param data: data to send to the cluster
        :return: aggregated data from the all the nodes in the cluster
        """
        # For single node or when collective is not initialized, just return the data
        if self.world_size == 1:
            return [data]

        try:
            collective.get_rank()  # Test if collective is initialized
        except Exception:
            logging.error("collective get_rank failed", exc_info=True)
            return [data]

        results = []
        data_str = json.dumps(data)
        for i in range(self.world_size):
            if self.rank == i:
                logging.info("Broadcasting data from self ({}) to others".format(self.rank))
                collective.broadcast(data_str, i)
                results.append(data)
            else:
                logging.info("Receiving data from {}".format(i))
                message_str = collective.broadcast("", i)
                message = json.loads(message_str) if message_str else None
                results.append(message)
        return results


class Rabit(object):
    @staticmethod
    def _get_logger(current_host):
        logging.basicConfig(format="%(name) [{}]: %(message)s".format(current_host))
        return logging.getLogger("RabitContextManager")

    def __init__(
        self, hosts, current_host=None, master_host=None, port=None, max_connect_attempts=None, connect_retry_timeout=3
    ):
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
        if not current_host:
            current_host = LOCAL_HOSTNAME
        self.current_host = current_host
        self.logger = self._get_logger(self.current_host)
        self.logger.debug("Found current host.")

        self.hosts = sorted(hosts)
        self.n_workers = len(self.hosts)
        self.logger.debug("Found hosts: {} [{}]".format(self.hosts, self.n_workers))

        if not master_host:
            master_host = self.hosts[0]
        self.master_host = master_host
        self.is_master_host = self.current_host == self.master_host

        self.logger.debug("Is Master: {}".format(self.is_master_host))
        self.logger.debug("Master: {}".format(self.master_host))

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
        """Start the collective process.

        Initialize XGBoost collective for distributed training.

        :return: Initialized RabitHelper, which includes helpful information such as is_master and port
        """
        self.logger.debug("Starting collective communication.")
        self.tracker = None
        self.tracker_thread = None

        # For single node, skip collective initialization
        if self.n_workers == 1:
            self.logger.debug("Single worker detected, skipping collective init")
            return RabitHelper(True, self.current_host, self.port)

        try:
            # Launch tracker on master only
            if self.is_master_host:
                self.tracker = RabitTracker(
                    host_ip=str(_dns_lookup(self.master_host)),
                    n_workers=int(self.n_workers),
                    port=int(self.port),
                    sortby="task",
                )
                self.tracker.start()
                self.tracker_thread = Thread(target=self.tracker.wait_for)
                self.tracker_thread.daemon = True
                self.tracker_thread.start()
                self.logger.info(f"RabitTracker worker_args: {self.tracker.worker_args()}")

            self.logger.info(
                f"MASTER_DEBUG_FIXED: Using hostname logic: \
                    current_host={self.current_host}, \
                        master_host={self.master_host}, \
                            is_master={self.is_master_host}, \
                                port={self.port}"
            )

            import time

            attempt = 0
            successful_connection = False
            while not successful_connection and (
                self.max_connect_attempts is None or attempt < self.max_connect_attempts
            ):
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
                raise Exception(f"Failed to connect to Rabit Tracker, current_host={self.current_host}")
            else:
                self.logger.info(f"Connected to RabitTracker, current_host={self.current_host}")

            # Initialize collective for synchronization
            collective.init(
                dmlc_tracker_uri=str(_dns_lookup(self.master_host)),
                dmlc_tracker_port=int(self.port),
                dmlc_task_id=str(self.hosts.index(self.current_host)),
                dmlc_retry=self.max_connect_attempts,
                dmlc_timeout=self.connect_retry_timeout,
            )

        except Exception as e:
            self.logger.error(f"{self.current_host} collective init failed", exc_info=True)
            self._cleanup_tracker()
            raise e

        self.logger.info(f"RABIT_START_DEBUG: Creating RabitHelper with is_master={self.is_master_host}")
        return RabitHelper(self.is_master_host, self.current_host, self.port)

    def stop(self):
        """Shutdown collective communication."""
        self.logger.info(f"Shutting down collective, current_host={self.current_host}")

        try:
            collective.finalize()
        except Exception:
            self.logger.error(f"{self.current_host} collective finalize failed", exc_info=True)

        # Wait for tracker thread to finish
        if self.tracker_thread is not None:
            try:
                self.tracker_thread.join(timeout=1.0)
            except Exception as e:
                self.logger.debug("Tracker thread join failed: {}".format(e))
            finally:
                self.tracker_thread = None

        self._cleanup_tracker()

    def _cleanup_tracker(self):
        """Clean up tracker safely."""
        if self.tracker is not None:
            try:
                self.tracker.free()
            except Exception as e:
                self.logger.debug("Tracker cleanup failed: {}".format(e))
            finally:
                self.tracker = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()
