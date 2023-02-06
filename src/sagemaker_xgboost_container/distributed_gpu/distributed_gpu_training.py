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
import time
from typing import Dict

import xgboost as xgb
from dask.distributed import Client

from sagemaker_algorithm_toolkit import exceptions as exc
from sagemaker_xgboost_container.constants.xgb_constants import MODEL_NAME
from sagemaker_xgboost_container.distributed_gpu.dask_cluster_utils import (
    get_host_ip,
    start_daemons_in_current_instance,
)
from sagemaker_xgboost_container.distributed_gpu.dask_data_utils import (
    create_dask_dmatrix,
    get_dataframe_dimensions,
    load_data_into_memory,
)

logger = logging.getLogger(__name__)

SCHEDULER_PORT = "8786"
WAIT_FOR_ALL_WORKERS_TIMEOUT_SEC = 20
WORKER_STAY_ALIVE_CHECK_FREQ_SEC = 10


def run_training_with_dask(
    hyperparameters: Dict,
    train_path: str,
    validation_path: str,
    model_dir: str,
    content_type: str,
    sm_hosts: [str],
    current_host: str,
    num_gpus: int,
):
    scheduler_host = sm_hosts[0]
    scheduler_host_ip = get_host_ip(scheduler_host)

    scheduler_address = f"{scheduler_host_ip}:{SCHEDULER_PORT}"
    is_scheduler_host = current_host == scheduler_host

    start_daemons_in_current_instance(scheduler_address, is_scheduler_host)

    total_num_workers = len(sm_hosts) * num_gpus

    # We only need to submit the job from one instance.
    if is_scheduler_host:
        with Client(scheduler_address) as client:
            # We ensure that all workers are present before proceeding.
            client.wait_for_workers(total_num_workers, WAIT_FOR_ALL_WORKERS_TIMEOUT_SEC)

            logging.info("Starting to read training data...")
            watchlist = []

            X_train, y_train = load_data_into_memory(client, train_path, content_type)

            dtrain = create_dask_dmatrix(client, X_train, y_train)

            # Log train data dimension for sanity check.
            train_num_rows, train_num_cols = get_dataframe_dimensions(X_train)
            logging.info(f"Train features matrix has {train_num_rows} rows and {train_num_cols} columns")

            watchlist.append((dtrain, "train"))

            if validation_path is not None:
                X_valid, y_valid = load_data_into_memory(client, validation_path, content_type)
                dvalid = create_dask_dmatrix(client, X_valid, y_valid)
                watchlist.append((dvalid, "validation"))

            logging.info("Data load complete. Starting training...")

            try:
                output = xgb.dask.train(
                    client, hyperparameters, dtrain, num_boost_round=hyperparameters["num_round"], evals=watchlist
                )
                booster = output["booster"]

                logging.info("Training complete. Saving model...")
                booster.save_model(os.path.join(model_dir, MODEL_NAME))
            except Exception as e:
                exception_prefix = "XGB train call failed with exception"
                raise exc.AlgorithmError(f"{exception_prefix}:\n {str(e)}")

            logging.info("Terminating cluster...")

    else:
        scheduler = (scheduler_host_ip, int(SCHEDULER_PORT))
        # Do not exit till the job is done.
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as alive_socket:
                alive_check = alive_socket.connect_ex(scheduler)
                if alive_check != 0:
                    logging.info("Received a shutdown signal from scheduler. Exiting...")
                    break
            time.sleep(WORKER_STAY_ALIVE_CHECK_FREQ_SEC)
