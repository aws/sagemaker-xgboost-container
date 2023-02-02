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

import logging
import os
import socket
from subprocess import Popen
import sys
import time

import dask.dataframe as dask_dataframe
from dask.dataframe import DataFrame
from dask.dataframe import Series
from dask.distributed import Client, wait
from sagemaker_algorithm_toolkit import exceptions as exc
import xgboost as xgb

from sagemaker_xgboost_container.constants.xgb_constants import MODEL_NAME

DASK_PATH = "/miniconda3/bin"
SCHEDULER_PORT = "8786"

WAIT_TIME_DAEMONS_STARTUP_SEC = 2
WAIT_FOR_ALL_WORKERS_TIMEOUT_SEC = 20

logger = logging.getLogger(__name__)


def _start_daemons(scheduler_host_ip: str, current_host_ip: str):
    cmd_start_scheduler = os.path.join(DASK_PATH, "dask-scheduler")
    cmd_start_worker = os.path.join(DASK_PATH, "dask-cuda-worker")
    schedule_conn_string = "tcp://{}:{}".format(scheduler_host_ip, SCHEDULER_PORT)

    # We can set this based on number of vCPUs in the future.
    # So far haven't seen improvement with > 4, during testing.
    num_threads_per_worker_process = str(4)
    # Dask cuda worker API doc: https://docs.rapids.ai/api/dask-cuda/nightly/api.html
    worker_cli_command = [cmd_start_worker, schedule_conn_string, "--no-dashboard",
                          "--nthreads", num_threads_per_worker_process]
    # Dask distributed scheduler API doc: https://docs.dask.org/en/stable/deploying-cli.html
    scheduler_cli_command = [cmd_start_scheduler, "--no-dashboard"]
    if scheduler_host_ip == current_host_ip:
        Popen(scheduler_cli_command)
    Popen(worker_cli_command)


def _get_host_ip(host_name: str) -> str:
    ip_wait_time = 5
    counter = 0
    host_ip = ""
    from socket import gaierror
    while counter < ip_wait_time and host_ip == "":
        try:
            host_ip = socket.gethostbyname(host_name)
            break
        except gaierror:
            counter += 1
            time.sleep(1)

    if counter == ip_wait_time and host_ip == "":
        raise Exception(
            "Exceeded max wait time of {}s for hostname resolution for host {}.".format(ip_wait_time, host_name)
        )
    return host_ip


def _read_data(client: Client, local_path: str, content_type: str) -> (DataFrame, Series):
    if content_type == "csv":
        dataframe = dask_dataframe.read_csv(local_path + "/*.csv", header=None)
    # content_type should only be csv or parquet after checks in train.py.
    else:
        dataframe = dask_dataframe.read_parquet(local_path)
    target_column = dataframe.columns[0]
    labels = dataframe[target_column]
    features = dataframe[dataframe.columns.difference([target_column])]

    features, labels = client.persist(
        [features, labels]
    )
    wait([features, labels])

    return features, labels


def run_training_with_dask(
        hyperparameters: {},
        train_path: str,
        validation_path: str,
        model_dir: str,
        content_type: str,
        sm_hosts: [str],
        current_host: str,
        num_gpus: int,
):
    current_host_ip = _get_host_ip(current_host)
    scheduler_host_ip = _get_host_ip(sm_hosts[0])

    _start_daemons(scheduler_host_ip, current_host_ip)

    total_num_workers = len(sm_hosts) * num_gpus
    is_scheduler_host = current_host == sm_hosts[0]

    time.sleep(WAIT_TIME_DAEMONS_STARTUP_SEC)

    # We only need to submit the job from one node/container.
    if is_scheduler_host:
        scheduler_address = "{}:{}".format(scheduler_host_ip, SCHEDULER_PORT)
        with Client(scheduler_address) as client:
            # We ensure that all workers are present before proceeding.
            client.wait_for_workers(total_num_workers, WAIT_FOR_ALL_WORKERS_TIMEOUT_SEC)

            logging.info("Starting to read training data...")
            watchlist = []
            try:
                X_train, y_train = _read_data(client, train_path, content_type)

                dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)

                # X_train.shape[0].compute() is an expensive operation, but provides sanity check.
                logging.info(
                    "Train matrix has {} rows and {} columns".format(X_train.shape[0].compute(), len(X_train.columns)))
                watchlist.append((dtrain, 'train'))

                if validation_path is not None:
                    X_valid, y_valid = _read_data(client, validation_path, content_type)
                    dvalid = xgb.dask.DaskDMatrix(client, X_valid, y_valid)
                    watchlist.append((dvalid, 'validation'))

            except Exception as e:
                raise exc.UserError("Failed to load data with exception:\n{}".format(e))
                return

            logging.info("Data load complete. Starting training...")

            try:
                output = xgb.dask.train(client,
                                        hyperparameters,
                                        dtrain,
                                        num_boost_round=hyperparameters["num_round"],
                                        evals=watchlist)
                booster = output['booster']

                logging.info("Training complete. Saving model...")
                booster.save_model(os.path.join(model_dir, MODEL_NAME))
            except Exception as e:
                exception_prefix = "XGB train call failed with exception"
                raise exc.AlgorithmError("{}:\n {}".format(exception_prefix, str(e)))

            logging.info("Terminating cluster...")

    else:
        # Do not exit till the job is done.
        while True:
            scheduler = (scheduler_host_ip, int(SCHEDULER_PORT))
            alive_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            alive_check = alive_socket.connect_ex(scheduler)
            if alive_check == 0:
                pass
            else:
                logging.info("Received a shutdown signal from scheduler. Exiting...")
                sys.exit(0)
            alive_socket.close()
            time.sleep(2)
    return
