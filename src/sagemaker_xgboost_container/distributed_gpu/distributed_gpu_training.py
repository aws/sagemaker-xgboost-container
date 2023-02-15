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
from sagemaker_algorithm_toolkit.channel_validation import S3_DIST_TYPE, Channel
from sagemaker_algorithm_toolkit.exceptions import UserError
from sagemaker_xgboost_container.algorithm_mode import train_utils
from sagemaker_xgboost_container.callback import get_callbacks
from sagemaker_xgboost_container.constants.xgb_constants import (
    GPU_TREE_METHOD,
    MODEL_NAME,
    PIPE_MODE,
)
from sagemaker_xgboost_container.data_utils import CSV, PARQUET
from sagemaker_xgboost_container.distributed_gpu.dask_cluster_utils import (
    get_host_ip,
    start_daemons_in_current_instance,
)
from sagemaker_xgboost_container.distributed_gpu.dask_data_utils import (
    create_dask_dmatrix,
    get_dataframe_dimensions,
    read_data,
)

logger = logging.getLogger(__name__)

SCHEDULER_PORT = "8786"
WAIT_FOR_ALL_WORKERS_TIMEOUT_SEC = 20
WORKER_STAY_ALIVE_CHECK_FREQ_SEC = 10
SUPPORTED_TRAINING_CONTENT_TYPES = {CSV, PARQUET}

NON_GPU_ERROR_MSG = "Dask training is only available for `gpu_hist` training on GPU instances."
PIPE_MODE_ERROR_MSG = "Dask training is not supported for pipe mode input."
INPUT_FORMAT_ERROR_MSG = "Dask training is only supported for CSV and Parquet input."
NOT_REPLICATED_ERROR_MSG = "Dask distributed training requires FullyReplicated data."


def check_if_all_conditions_met(
    tree_method_hp: str, num_hosts: int, num_gpus: int, input_mode: str, input_format: str, data_config: Dict
):
    if tree_method_hp != GPU_TREE_METHOD or num_gpus == 0:
        raise UserError(NON_GPU_ERROR_MSG)
    if input_mode == PIPE_MODE:
        raise UserError(PIPE_MODE_ERROR_MSG)
    if input_format not in SUPPORTED_TRAINING_CONTENT_TYPES:
        raise UserError(INPUT_FORMAT_ERROR_MSG)
    is_channels_not_replicated = any(
        {channel.get(S3_DIST_TYPE, None) != Channel.REPLICATED for channel in data_config.values()}
    )
    if is_channels_not_replicated and num_hosts > 1:
        raise UserError(NOT_REPLICATED_ERROR_MSG)


def run_training_with_dask(
    hyperparameters: Dict,
    train_path: str,
    validation_path: str,
    model_dir: str,
    content_type: str,
    sm_hosts: [str],
    current_host: str,
    checkpoint_dir: str,
    num_gpus: int,
):
    scheduler_host = sm_hosts[0]
    is_scheduler_host = current_host == scheduler_host
    scheduler_host_ip = get_host_ip(scheduler_host)

    scheduler_address = f"{scheduler_host_ip}:{SCHEDULER_PORT}"
    start_daemons_in_current_instance(scheduler_address, is_scheduler_host)

    total_num_workers = len(sm_hosts) * num_gpus

    # We only need to submit the job from one instance.
    if is_scheduler_host:
        with Client(scheduler_address) as client:
            # We ensure that all workers are present before proceeding.
            client.wait_for_workers(total_num_workers, WAIT_FOR_ALL_WORKERS_TIMEOUT_SEC)

            logging.info("Starting to read training data...")
            watchlist = []

            X_train, y_train = read_data(train_path, content_type)

            dtrain = create_dask_dmatrix(client, X_train, y_train)

            # Log train data dimension for sanity check.
            train_num_rows, train_num_cols = get_dataframe_dimensions(X_train)
            logging.info(f"Train features matrix has {train_num_rows} rows and {train_num_cols} columns")

            watchlist.append((dtrain, "train"))

            dvalid = None
            if validation_path:
                X_valid, y_valid = read_data(validation_path, content_type)
                dvalid = create_dask_dmatrix(client, X_valid, y_valid)
                watchlist.append((dvalid, "validation"))

            logging.info("Data load complete. Starting training...")

            # Redundant Code -------------------------------------------------------------------- >
            """
            The blob below is the redundant code we can see in the original training flow. Some
            points that might be worth addressing in the copied blob below are as follows:
            * Are hardcoded static string references everywhere really the best way to do this?
            * Can we consolidate this data cleanup and popping business into an appropriate class?
            * Similar to the above point, is an anemic data model/similar an angle worth considering?
            * Do we have overhead concerns with the debugging if we need to convert to DMatrix?
            * Does allowing for cross validation outweigh overhead concerns between CPU & GPU?
            """
            num_round = hyperparameters.pop("num_round")
            save_model_on_termination = hyperparameters.pop("save_model_on_termination", "false")
            tuning_objective_metric_param = hyperparameters.pop("_tuning_objective_metric", None)
            eval_metric = hyperparameters.pop("eval_metric", None)
            cleaned_eval_metric, configured_feval, tuning_objective_metric = train_utils.get_eval_metrics_and_feval(
                tuning_objective_metric_param, eval_metric
            )
            if cleaned_eval_metric:
                hyperparameters["eval_metric"] = cleaned_eval_metric

            early_stopping_data_name = "validation" if dvalid else None
            early_stopping_rounds = hyperparameters.pop("early_stopping_rounds", None)

            early_stopping_metric = None
            if early_stopping_rounds:
                if tuning_objective_metric:
                    early_stopping_metric = tuning_objective_metric[-1]
                elif eval_metric:
                    early_stopping_metric = eval_metric[-1]

            xgb_model, iteration, callbacks = get_callbacks(
                model_dir=model_dir,
                checkpoint_dir=checkpoint_dir,
                early_stopping_data_name=early_stopping_data_name,
                early_stopping_metric=early_stopping_metric,
                early_stopping_rounds=early_stopping_rounds,
                save_model_on_termination=save_model_on_termination,
                is_master=is_scheduler_host,
            )

            try:
                output = xgb.dask.train(
                    client=client,
                    params=hyperparameters,
                    dtrain=dtrain,
                    num_boost_round=num_round,
                    evals=watchlist,
                    feval=configured_feval,
                    callbacks=callbacks,
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
