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

import socket
from subprocess import Popen

from dask.distributed import Client

from sagemaker_algorithm_toolkit.exceptions import AlgorithmError, PlatformError

SCHEDULER_EXEC_PATH = "/miniconda3/bin/dask-scheduler"
CUDA_WORKER_EXEC_PATH = "/miniconda3/bin/dask-cuda-worker"

SCHEDULER_CONN_TIMEOUT = "20s"


def start_daemons_in_current_instance(scheduler_address: str, is_scheduler_host: bool):
    # Dask distributed scheduler API doc: https://docs.dask.org/en/stable/deploying-cli.html
    scheduler_cli_command = [SCHEDULER_EXEC_PATH, "--no-dashboard"]
    scheduler_conn_string = f"tcp://{scheduler_address}"
    # Dask cuda worker API doc: https://docs.rapids.ai/api/dask-cuda/nightly/api.html
    worker_cli_command = [CUDA_WORKER_EXEC_PATH, scheduler_conn_string, "--no-dashboard"]
    if is_scheduler_host:
        Popen(scheduler_cli_command)
    try:
        # Ensure that the scheduler is up before starting workers.
        with Client(scheduler_address, timeout=SCHEDULER_CONN_TIMEOUT):
            Popen(worker_cli_command)
    except TimeoutError as e:
        raise AlgorithmError(
            f"Couldn't connect to scheduler after {SCHEDULER_CONN_TIMEOUT}. Please try re-running the training job."
            f" Exception: {e}"
        )


def get_host_ip(host_name: str) -> str:
    try:
        host_ip = socket.gethostbyname(host_name)
    except socket.gaierror as e:
        # This shouldn't have happened, and it's not the user's fault.
        raise PlatformError(f"Failed hostname resolution for host '{host_name}', exception: {e}")
    return host_ip
