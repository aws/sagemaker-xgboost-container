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
import time
import threading
from typing import List, Dict, Any, Optional

import numpy as np
from retrying import retry

# XGBoost 2.1.0 uses collective communication instead of rabit
try:
    import xgboost as xgb
    from xgboost import collective
    from xgboost.collective import CommunicatorContext
except ImportError:
    raise ImportError("XGBoost 2.1.0 or later is required")

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
    connect_retry_timeout=3,
    update_rabit_args=False,
):
    """Run execution function after initializing xgboost collective communication.

    This method initializes collective communication twice:
        1. To broadcast to all hosts which hosts should be included in training.
        2. Run distributed xgb train() with just the hosts from above.

    :param exec_fun: Function to run while collective is initialized. xgb.train() must run in the same process space
                    in order to utilize collective initialization. Note that the execution function must also take the args
                    'is_distributed' and 'is_master'.
    :param args: Arguments to run execution function.
    :param include_in_training: Boolean if the current hosts should be used in training. This is done here so that
                                all the hosts in the cluster know which hosts to include during training.
    :param hosts:
    :param current_host:
    :param first_port: Port to use for the initial collective initialization. If None, defaults to 9099
    :param second_port: Port to use for second collective initialization. If None, this increments previous port by 1
    :param max_connect_attempts
    :param connect_retry_timeout
    :param update_rabit_args: Boolean to include collective information to args. If True, the following is added:
                                is_master
    """
    with Rabit(
        hosts=hosts,
        current_host=current_host,
        port=first_port,
        max_connect_attempts=max_connect_attempts,
        connect_retry_timeout=connect_retry_timeout,
    ) as rabit_helper:
        hosts_with_data = rabit_helper.synchronize({"host": rabit_helper.current_host, "include_in_training": include_in_training})
        hosts_with_data = [record["host"] for record in hosts_with_data if record["include_in_training"]]

        # Keep track of port used, so that hosts trying to shutdown know when server is not available
        previous_port = rabit_helper.master_port

    if not include_in_training:
        logging.warning("Host {} not being used for distributed training.".format(current_host))
        sys.exit(0)

    second_rabit_port = second_port if second_port else previous_port + 1

    if len(hosts_with_data) > 1:
        # Set up collective with nodes that have data and an unused port so that previous slaves don't confuse it
        # with the previous collective configuration
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
        logging.debug(
            "Only 1 host with training data, " "starting single node training job from: {}".format(current_host)
        )
        if update_rabit_args:
            args.update({"is_master": True})
        exec_fun(**args)

    else:
        raise RuntimeError("No hosts received training data.")


class RabitHelper(object):
    def __init__(self, is_master, current_host, master_port, is_collective_initialized=False):
        """This is returned by the Rabit context manager for useful cluster information and data synchronization.

        :param is_master:
        :param current_host:
        :param master_port:
        :param is_collective_initialized: Whether XGBoost collective communication is initialized
        """
        self.is_master = is_master
        self.current_host = current_host
        self.master_port = master_port
        self._is_collective_initialized = is_collective_initialized
        
        if is_collective_initialized:
            self.rank = collective.get_rank()
        else:
            self.rank = 0

    def tracker_print(self, msg: str):
        """Print message to tracker log.
        
        Equivalent to rabit.tracker_print() - prints a message to the centralized 
        logging facility for tracking progress across the distributed cluster.
        
        :param msg: Message to print to tracker log
        """
        if self._is_collective_initialized:
            # Use collective.print for distributed case
            collective.print(msg)
        else:
            # For single node case, just use regular logging
            logging.info(f"[Tracker] {msg}")

    def synchronize(self, data):
        """Synchronize data with the cluster.

        This function allows every node to share state with every other node easily.
        This allows things like determining which nodes have data or not.

        :param data: data to send to the cluster
        :return: aggregated data from all the nodes in the cluster
        """
        if not self._is_collective_initialized:
            # Single node case
            return [data]
            
        results = []
        world_size = collective.get_world_size()
        
        for i in range(world_size):
            if self.rank == i:
                logging.debug("Broadcasting data from self ({}) to others".format(self.rank))
                collective.broadcast(data, i)
                results.append(data)
            else:
                logging.debug("Receiving data from {}".format(i))
                message = collective.broadcast(None, i)
                results.append(message)
        return results


class SimpleTracker:
    """Simple tracker implementation for XGBoost collective communication"""
    
    def __init__(self, host_ip: str, n_workers: int, port: int):
        self.host_ip = host_ip
        self.n_workers = n_workers
        self.port = port
        self.server_socket = None
        self.server_thread = None
        self._shutdown = threading.Event()
        
    def start(self, n_workers: int):
        """Start the tracker server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host_ip, self.port))
        self.server_socket.listen(n_workers)
        self.server_socket.settimeout(1.0)  # Non-blocking accept
        
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()
        
    def _server_loop(self):
        """Simple server loop to accept connections"""
        connected_clients = 0
        while not self._shutdown.is_set() and connected_clients < self.n_workers:
            try:
                client_socket, addr = self.server_socket.accept()
                connected_clients += 1
                logging.debug(f"Tracker accepted connection from {addr}")
                client_socket.close()
            except socket.timeout:
                continue
            except Exception as e:
                if not self._shutdown.is_set():
                    logging.debug(f"Tracker server error: {e}")
                break
                
    def join(self):
        """Wait for the server thread to finish"""
        self._shutdown.set()
        if self.server_thread:
            self.server_thread.join()
        if self.server_socket:
            self.server_socket.close()
            
    def slave_envs(self):
        """Return environment configuration for slaves"""
        return {
            'DMLC_NUM_WORKER': str(self.n_workers),
            'DMLC_TRACKER_URI': self.host_ip,
            'DMLC_TRACKER_PORT': str(self.port)
        }


class Rabit(object):
    @staticmethod
    def _get_logger(current_host):
        logging.basicConfig(format="%(name)s [{}]: %(message)s".format(current_host))
        return logging.getLogger("RabitContextManager")

    def __init__(
        self, 
        hosts: List[str], 
        current_host: Optional[str] = None, 
        master_host: Optional[str] = None, 
        port: Optional[int] = None, 
        max_connect_attempts: Optional[int] = None, 
        connect_retry_timeout: int = 3
    ):
        """Context manager for XGBoost collective communication initialization.

        :param hosts: List of hostnames
        :param current_host: Current hostname. If not provided, use 127.0.0.1.
        :param master_host: Master host hostname. If not provided, use alphabetically first hostname amongst hosts
                            to ensure determinism in choosing master node.
        :param port: Port to connect to master, if not specified use 9099.
        :param max_connect_attempts: Number of times to try connecting to tracker. If this arg is set
                            to None, try indefinitely.
        :param connect_retry_timeout: Timeout value when attempting to connect to tracker.
                            This will be ignored if max_connect_attempt is None
        """
        # Get the host information
        if not current_host:
            current_host = LOCAL_HOSTNAME
        self.current_host = current_host
        self.logger = self._get_logger(self.current_host)
        self.logger.debug("Found current host.")

        self.hosts = sorted(hosts)
        self.n_workers = len(self.hosts)
        self.logger.debug("Found hosts: {} [{}]".format(self.hosts, self.n_workers))

        # Use the first lexicographically named host as the master if not indicated otherwise
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
        
        self.tracker = None

    def start(self):
        """Start the collective communication process.

        If current host is master host, initialize and start the tracker in the background. 
        All hosts then connect to the master host to set up collective communication.

        :return: Initialized RabitHelper, which includes helpful information such as is_master and port
        """
        if self.n_workers == 1:
            # Single node case - no need for collective communication
            self.logger.debug("Single node training - skipping collective communication setup")
            return RabitHelper(True, self.current_host, self.port)
            
        if self.is_master_host:
            self.logger.debug("Master host. Starting Tracker.")
            self.tracker = SimpleTracker(
                host_ip=self.current_host,
                n_workers=self.n_workers,
                port=self.port
            )
            self.logger.info("Tracker slave environment: {}".format(self.tracker.slave_envs()))
            self.tracker.start(self.n_workers)

        # Start parameter server that connects to the master
        self.logger.debug("Starting parameter server.")

        # Wait for tracker to be available
        attempt = 0
        successful_connection = False
        while not successful_connection and (self.max_connect_attempts is None or attempt < self.max_connect_attempts):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    self.logger.debug("Checking if Tracker is available.")
                    s.connect((self.master_host, self.port))
                    successful_connection = True
                    self.logger.debug("Successfully connected to Tracker.")
                except OSError:
                    self.logger.info("Failed to connect to Tracker on attempt {}".format(attempt))
                    attempt += 1
                    self.logger.info("Sleeping for {} sec before retrying".format(self.connect_retry_timeout))
                    time.sleep(self.connect_retry_timeout)

        if not successful_connection:
            self.logger.error("Failed to connect to Tracker after %s attempts", self.max_connect_attempts)
            raise Exception("Failed to connect to Tracker")
        else:
            self.logger.info("Connected to Tracker.")

        # Initialize XGBoost collective communication using environment variables
        import os
        os.environ['DMLC_NUM_WORKER'] = str(self.n_workers)
        os.environ['DMLC_TRACKER_URI'] = self.master_host
        os.environ['DMLC_TRACKER_PORT'] = str(self.port)
        
        # Initialize collective communication
        collective.init()

        # Get rank information
        rank = collective.get_rank() if self.n_workers > 1 else 0
        self.logger.debug("Collective started - Rank {}".format(rank))
        self.logger.debug("Executing user code")

        return RabitHelper(self.is_master_host, self.current_host, self.port, True)

    def stop(self):
        """Shutdown parameter server and tracker."""
        self.logger.debug("Shutting down parameter server.")

        # Clean up collective communication
        if self.n_workers > 1:
            try:
                collective.finalize()
            except Exception as e:
                self.logger.debug(f"Error finalizing collective: {e}")
            
        if self.is_master_host and self.tracker:
            self.tracker.join()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()
        return False