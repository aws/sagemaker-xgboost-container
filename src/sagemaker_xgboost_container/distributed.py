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
Updated for XGBoost 3.x collective API.
"""
import json
import logging
import socket
import sys
import time

from retrying import retry
from xgboost import collective
from xgboost.tracker import RabitTracker

LOCAL_HOSTNAME = "127.0.0.1"


@retry(stop_max_delay=1000 * 60 * 15, wait_exponential_multiplier=100, wait_exponential_max=30000)
def _dns_lookup(host):
    """Retrying dns lookup on host"""
    return socket.gethostbyname(host)


def wait_hostname_resolution(sm_hosts):
    """Wait for the hostname resolution of the container."""
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
    connect_retry_timeout=10,
    update_rabit_args=False,
):
    """Run execution function after initializing rabit.

    This method initializes rabit twice:
        1. To broadcast to all hosts which hosts should be included in training.
        2. Run distributed xgb train() with just the hosts from above.

    :param exec_fun: Function to run while rabit is initialized.
    :param args: Arguments to run execution function.
    :param include_in_training: Boolean if the current host should be used in training.
    :param hosts: List of hostnames.
    :param current_host: Current hostname.
    :param first_port: Port for initial rabit initialization. Defaults to 9099.
    :param second_port: Port for second rabit initialization.
    :param max_connect_attempts: Number of times to try connecting.
    :param connect_retry_timeout: Timeout between connection attempts.
    :param update_rabit_args: Boolean to include rabit information to args.
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
        previous_port = rabit_ctx.master_port

    if not include_in_training:
        logging.warning("Host {} not being used for distributed training.".format(current_host))
        sys.exit(0)

    second_rabit_port = second_port if second_port else previous_port + 1

    if len(hosts_with_data) > 1:
        with Rabit(
            hosts=hosts_with_data,
            current_host=current_host,
            port=second_rabit_port,
            max_connect_attempts=max_connect_attempts,
            connect_retry_timeout=connect_retry_timeout,
        ) as cluster:
            if update_rabit_args:
                args.update({"is_master": cluster.is_master})
            exec_fun(**args)

    elif len(hosts_with_data) == 1:
        logging.debug("Only 1 host with training data, starting single node training.")
        if update_rabit_args:
            args.update({"is_master": True})
        exec_fun(**args)

    else:
        raise RuntimeError("No hosts received training data.")


class RabitHelper(object):
    def __init__(self, is_master, current_host, master_port):
        self.is_master = is_master
        self.current_host = current_host
        self.master_port = master_port

        try:
            self.rank = collective.get_rank()
            self.world_size = collective.get_world_size()
        except Exception:
            self.rank = 0
            self.world_size = 1

    def synchronize(self, data):
        """Synchronize data across all workers using collective broadcast."""
        if self.world_size == 1:
            return [data]

        results = []
        data_str = json.dumps(data)
        for i in range(self.world_size):
            if self.rank == i:
                result = str(collective.broadcast(data_str, i))
            else:
                result = str(collective.broadcast("", i))
            results.append(json.loads(result))
        return results


class Rabit(object):
    @staticmethod
    def _get_logger(current_host):
        logging.basicConfig(format="%(name)s [{}]: %(message)s".format(current_host))
        return logging.getLogger("RabitContextManager")

    def __init__(
        self, hosts, current_host=None, master_host=None, port=None, max_connect_attempts=None, connect_retry_timeout=3
    ):
        if not current_host:
            current_host = LOCAL_HOSTNAME
        self.current_host = current_host
        self.logger = self._get_logger(self.current_host)

        self.hosts = sorted(hosts)
        self.n_workers = len(self.hosts)

        if not master_host:
            master_host = self.hosts[0]
        self.master_host = master_host
        self.is_master_host = self.current_host == self.master_host

        if port is None:
            port = 9099
        self.port = port

        if max_connect_attempts is not None and max_connect_attempts <= 0:
            raise ValueError("max_connect_attempts must be None or an integer greater than 0.")
        self.max_connect_attempts = max_connect_attempts
        self.connect_retry_timeout = connect_retry_timeout

    def start(self):
        """Start collective using XGBoost 3.x CommunicatorContext."""
        self.tracker = None
        self._comm_ctx = None

        if self.n_workers == 1:
            return RabitHelper(True, self.current_host, self.port)

        try:
            # Launch tracker on master
            if self.is_master_host:
                self.tracker = RabitTracker(
                    n_workers=self.n_workers,
                    host_ip=str(_dns_lookup(self.master_host)),
                    port=self.port,
                    sortby="task",
                )
                self.tracker.start()
                self._worker_args = self.tracker.worker_args()
                self.logger.info(f"RabitTracker started, worker_args: {self._worker_args}")
            else:
                self._worker_args = None

            # Wait for tracker to be reachable
            self._wait_for_tracker()

            # Build worker args — non-master workers construct from known tracker info
            if self._worker_args is None:
                self._worker_args = {
                    "dmlc_tracker_uri": str(_dns_lookup(self.master_host)),
                    "dmlc_tracker_port": self.port,
                }

            # Add task ID for deterministic rank assignment
            self._worker_args["dmlc_task_id"] = str(self.hosts.index(self.current_host))

            if self.max_connect_attempts is not None:
                self._worker_args["dmlc_retry"] = self.max_connect_attempts
            if self.connect_retry_timeout is not None:
                self._worker_args["dmlc_timeout"] = self.connect_retry_timeout

            # Use CommunicatorContext for proper init/finalize
            self._comm_ctx = collective.CommunicatorContext(**self._worker_args)
            self._comm_ctx.__enter__()

        except Exception as e:
            self.logger.error(f"Collective init failed on {self.current_host}", exc_info=True)
            self._cleanup()
            raise e

        return RabitHelper(self.is_master_host, self.current_host, self.port)

    def _wait_for_tracker(self):
        """Wait for the tracker to become reachable."""
        attempt = 0
        while self.max_connect_attempts is None or attempt < self.max_connect_attempts:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.connect((self.master_host, self.port))
                    self.logger.debug("Connected to RabitTracker.")
                    return
                except OSError:
                    attempt += 1
                    time.sleep(self.connect_retry_timeout)

        raise Exception(f"Failed to connect to RabitTracker after {self.max_connect_attempts} attempts")

    def stop(self):
        """Shutdown collective communication."""
        if self._comm_ctx is not None:
            try:
                self._comm_ctx.__exit__(None, None, None)
            except Exception:
                self.logger.error("CommunicatorContext exit failed", exc_info=True)
            self._comm_ctx = None

        self._cleanup()

    def _cleanup(self):
        if self.tracker is not None:
            try:
                self.tracker.wait_for(timeout=5)
            except Exception:
                pass
            try:
                self.tracker.free()
            except Exception:
                pass
            self.tracker = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()
