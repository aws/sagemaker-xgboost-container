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

import numpy as np
import xgboost as xgb
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

from sagemaker_algorithm_toolkit import exceptions as exc
from sagemaker_algorithm_toolkit.channel_validation import Channel, S3_DIST_TYPE
from sagemaker_xgboost_container import distributed
from sagemaker_xgboost_container.algorithm_mode import channel_validation as cv
from sagemaker_xgboost_container.algorithm_mode import hyperparameter_validation as hpv
from sagemaker_xgboost_container.algorithm_mode import metrics as metrics_mod
from sagemaker_xgboost_container.algorithm_mode import train_utils
from sagemaker_xgboost_container.callback import add_debugging, get_callbacks
from sagemaker_xgboost_container.constants.sm_env_constants import SM_NUM_GPUS, SM_OUTPUT_DATA_DIR
from sagemaker_xgboost_container.constants.xgb_constants import CUSTOMER_ERRORS, MODEL_NAME
from sagemaker_xgboost_container.data_utils import (
    check_data_redundancy,
    get_content_type,
    get_dmatrix,
    get_size,
    validate_data_file_path,
)
from sagemaker_xgboost_container.distributed_gpu import distributed_gpu_training
from sagemaker_xgboost_container.prediction_utils import ValidationPredictionRecorder

logger = logging.getLogger(__name__)


def get_validated_dmatrices(
    train_path, validate_path, content_type, csv_weights=0, is_pipe=False, combine_train_val=False
):
    """Get training and validation Data Matrices for XGBoost training.

    Check size and format of both training and validation data channels, and return parsed
    Data Matrices.

    :param train_path:
    :param validate_path:
    :param content_type: Content type of data. Supports 'libsvm' or 'csv'
    :param csv_weights: 1 if instance weights are in the second column of csv data files; otherwise, 0
    :param is_pipe: Boolean to indicate if data is being read in pipe mode
    :combine_train_val: Boolean to indicate if returns a DMatrix combining train and validation data
    :return: Parsed xgb.DMatrix
    """
    train_files_size = get_size(train_path, is_pipe) if train_path else 0
    val_files_size = get_size(validate_path, is_pipe) if validate_path else 0

    if not is_pipe:
        logging.debug(
            "File size need to be processed in the node: {}mb.".format(
                round((train_files_size + val_files_size) / (1024 * 1024), 2)
            )
        )

        if train_files_size > 0:
            validate_data_file_path(train_path, content_type)
        if val_files_size > 0:
            validate_data_file_path(validate_path, content_type)

    train_dmatrix = (
        get_dmatrix(train_path, content_type, csv_weights=csv_weights, is_pipe=is_pipe)
        if train_files_size > 0
        else None
    )
    val_dmatrix = (
        get_dmatrix(validate_path, content_type, csv_weights=csv_weights, is_pipe=is_pipe)
        if val_files_size > 0
        else None
    )

    train_val_dmatrix = train_dmatrix
    if combine_train_val and train_dmatrix is not None and val_dmatrix is not None:
        logging.info("Read both train and validation data into one DMatrix")
        train_val_dmatrix = get_dmatrix(
            [train_path, validate_path], content_type, csv_weights=csv_weights, is_pipe=is_pipe
        )

    return train_dmatrix, val_dmatrix, train_val_dmatrix


def sagemaker_train(
    train_config, data_config, train_path, val_path, model_dir, sm_hosts, sm_current_host, checkpoint_config
):
    """Train XGBoost in a SageMaker training environment.

    Validate hyperparameters and data channel using SageMaker Algorithm Toolkit to fail fast if needed.
    If running with more than one host, check if the current host has data and run train_job() using
    rabit_run.

    :param train_config:
    :param data_config:
    :param train_path:
    :param val_path:
    :param model_dir:
    :param sm_hosts:
    :param sm_current_host:
    :param checkpoint_config:
    """
    metrics = metrics_mod.initialize()

    hyperparameters = hpv.initialize(metrics)
    validated_train_config = hyperparameters.validate(train_config)
    if validated_train_config.get("updater"):
        validated_train_config["updater"] = ",".join(validated_train_config["updater"])

    channels = cv.initialize()
    validated_data_config = channels.validate(data_config)

    logging.debug("hyperparameters {}".format(validated_train_config))
    logging.debug("channels {}".format(validated_data_config))

    # Get Training and Validation Data Matrices
    file_type = get_content_type(validated_data_config["train"].get("ContentType"))
    input_mode = validated_data_config["train"].get("TrainingInputMode")
    csv_weights = validated_train_config.get("csv_weights", 0)
    is_pipe = input_mode == Channel.PIPE_MODE

    validation_channel = validated_data_config.get("validation", None)
    combine_train_val = "_kfold" in validated_train_config
    if val_path is not None:
        if train_path == val_path or os.path.basename(train_path) == os.path.basename(val_path):
            logger.warning(
                "Found same path for training and validation. This is not recommended and results may not "
                "be correct."
            )
        elif not is_pipe:
            # Check if there is potential data redundancy between training and validation sets
            check_data_redundancy(train_path, val_path)

    # Obtain information about training resources to determine which distributed setup to use, if needed.
    num_hosts = len(sm_hosts)

    checkpoint_dir = checkpoint_config.get("LocalPath", None)

    num_gpus = int(os.getenv(SM_NUM_GPUS, 0))
    logging.info("Determined {} GPU(s) from environment variable 'SM_NUM_GPUS'".format(num_gpus))
    is_dask_job = num_gpus > 1 and train_config.get("tree_method", None) == "gpu_hist"
    if is_dask_job:
        # TODO: Following can likely be moved to hyperparamter validations
        is_unsupported_file_type = file_type not in distributed_gpu_training.SUPPORTED_TRAINING_CONTENT_TYPES
        is_channels_not_replicated = any({channel.get(S3_DIST_TYPE, None) != Channel.REPLICATED
                                          for channel in validated_data_config.values()})
        if is_unsupported_file_type or is_channels_not_replicated:
            raise exc.UserError("Multi-GPU training requires fully replicated data and only supports: {}".format(
                distributed_gpu_training.SUPPORTED_TRAINING_CONTENT_TYPES))
        logging.info("Going to run distributed training through Dask")
        distributed_gpu_training.run_training_with_dask(
            hyperparameters=validated_train_config,
            train_path=train_path,
            validation_path=val_path,
            model_dir=model_dir,
            content_type=file_type,
            sm_hosts=sm_hosts,
            current_host=sm_current_host,
            checkpoint_dir=checkpoint_dir,
            num_gpus=num_gpus)
    else:
        train_dmatrix, val_dmatrix, train_val_dmatrix = get_validated_dmatrices(
            train_path, val_path, file_type, csv_weights, is_pipe, combine_train_val)
        train_args = dict(
            train_cfg=validated_train_config,
            train_dmatrix=train_dmatrix,
            val_dmatrix=val_dmatrix,
            train_val_dmatrix=train_val_dmatrix,
            model_dir=model_dir,
            checkpoint_dir=checkpoint_dir)
        if num_hosts > 1:
            # Wait for hosts to find each other
            logging.info("Distributed node training with {} hosts: {}".format(num_hosts, sm_hosts))
            distributed.wait_hostname_resolution(sm_hosts)
            if not train_dmatrix:
                logging.warning(
                    "Host {} does not have data. Will broadcast to cluster and will not be used in distributed"
                    " training.".format(sm_current_host)
                )
            distributed.rabit_run(
                exec_fun=train_job,
                args=train_args,
                include_in_training=(train_dmatrix is not None),
                hosts=sm_hosts,
                current_host=sm_current_host,
                update_rabit_args=True)
        elif num_hosts == 1:
            if train_dmatrix:
                if validation_channel:
                    if not val_dmatrix:
                        raise exc.UserError("No data in validation channel path {}".format(val_path))
                logging.info("Single node training.")
                train_args.update({"is_master": True})
                train_job(**train_args)
            else:
                raise exc.UserError("No data in training channel path {}".format(train_path))
        else:
            raise exc.PlatformError("Number of hosts should be an int greater than or equal to 1")


def train_job(train_cfg, train_dmatrix, val_dmatrix, train_val_dmatrix, model_dir, checkpoint_dir, is_master):
    """Train and save XGBoost model using data on current node.

    If doing distributed training, XGBoost will use rabit to sync the trained model between each boosting iteration.
    Trained model is only saved if 'is_master' is True.

    :param train_cfg: Training hyperparameter configurations
    :param train_dmatrix: Training Data Matrix
    :param val_dmatrix: Validation Data Matrix
    :param train_val_dmatrix: Training + Validation Data Matrix
    :param model_dir: Directory where model will be saved
    :param is_master: True if single node training, or the current node is the master node in distributed training.
    """
    # Parse arguments for train() API
    num_round = train_cfg.pop("num_round")
    # Parse arguments for intermediate model callback
    save_model_on_termination = train_cfg.pop("save_model_on_termination", "false")

    # Evaluation metrics to use with train() API
    tuning_objective_metric_param = train_cfg.pop("_tuning_objective_metric", None)
    eval_metric = train_cfg.get("eval_metric")
    cleaned_eval_metric, configured_feval, tuning_objective_metric = train_utils.get_eval_metrics_and_feval(
        tuning_objective_metric_param, eval_metric
    )
    if cleaned_eval_metric:
        train_cfg["eval_metric"] = cleaned_eval_metric
    else:
        train_cfg.pop("eval_metric", None)

    early_stopping_rounds = train_cfg.pop("early_stopping_rounds", None)
    early_stopping_data_name = "validation" if val_dmatrix else None
    early_stopping_metric = None
    if early_stopping_rounds:
        if tuning_objective_metric:
            early_stopping_metric = tuning_objective_metric[-1]
        elif eval_metric:
            early_stopping_metric = eval_metric[-1]

    logging.info("Train matrix has {} rows and {} columns".format(train_dmatrix.num_row(), train_dmatrix.num_col()))
    if val_dmatrix:
        logging.info("Validation matrix has {} rows".format(val_dmatrix.num_row()))

    try:
        kfold = train_cfg.pop("_kfold", None)
        watchlist = [(train_dmatrix, "train")]
        if val_dmatrix is not None:
            watchlist.append((val_dmatrix, "validation"))

        if kfold is None:
            xgb_model, iteration, callbacks = get_callbacks(
                model_dir=model_dir,
                checkpoint_dir=checkpoint_dir,
                early_stopping_data_name=early_stopping_data_name,
                early_stopping_metric=early_stopping_metric,
                early_stopping_rounds=early_stopping_rounds,
                save_model_on_termination=save_model_on_termination,
                is_master=is_master,
            )
            add_debugging(
                callbacks=callbacks, hyperparameters=train_cfg, train_dmatrix=train_dmatrix, val_dmatrix=val_dmatrix
            )

            bst = xgb.train(
                train_cfg,
                train_dmatrix,
                num_boost_round=num_round - iteration,
                evals=watchlist,
                feval=configured_feval,
                callbacks=callbacks,
                xgb_model=xgb_model,
                verbose_eval=False,
            )

        else:
            num_cv_round = train_cfg.pop("_num_cv_round", 1)

            logging.info(
                "Run {}-round of {}-fold cross validation with {} rows".format(
                    num_cv_round, kfold, train_val_dmatrix.num_row()
                )
            )

            bst = []
            evals_results = []

            num_class = train_cfg.get("num_class", None)
            objective = train_cfg.get("objective", None)
            # RepeatedStratifiedKFold expects X as array-like of shape (n_samples, n_features)
            classification_problem = num_class or objective.startswith("binary:")
            num_rows_in_dataset = train_val_dmatrix.num_row()
            X = range(num_rows_in_dataset)
            y = train_val_dmatrix.get_label() if classification_problem else None
            rkf = (
                RepeatedStratifiedKFold(n_splits=kfold, n_repeats=num_cv_round)
                if y is not None
                else RepeatedKFold(n_splits=kfold, n_repeats=num_cv_round)
            )

            val_pred = ValidationPredictionRecorder(
                y_true=train_val_dmatrix.get_label(),
                num_cv_round=num_cv_round,
                classification=classification_problem,
                output_data_dir=os.environ[SM_OUTPUT_DATA_DIR],
            )
            for train_idx, val_idx in rkf.split(X=X, y=y):
                cv_train_dmatrix = train_val_dmatrix.slice(train_idx)
                cv_val_dmatrix = train_val_dmatrix.slice(val_idx)

                xgb_model, iteration, callbacks = get_callbacks(
                    model_dir=model_dir,
                    checkpoint_dir=checkpoint_dir,
                    early_stopping_data_name=early_stopping_data_name,
                    early_stopping_metric=early_stopping_metric,
                    early_stopping_rounds=early_stopping_rounds,
                    save_model_on_termination=save_model_on_termination,
                    is_master=is_master,
                    fold=len(bst),
                )
                add_debugging(
                    callbacks=callbacks,
                    hyperparameters=train_cfg,
                    train_dmatrix=cv_train_dmatrix,
                    val_dmatrix=cv_val_dmatrix,
                )

                evals_result = {}
                logging.info("Train cross validation fold {}".format((len(bst) % kfold) + 1))
                booster = xgb.train(
                    train_cfg,
                    cv_train_dmatrix,
                    num_boost_round=num_round - iteration,
                    evals=watchlist,
                    feval=configured_feval,
                    evals_result=evals_result,
                    callbacks=callbacks,
                    xgb_model=xgb_model,
                    verbose_eval=False,
                )
                bst.append(booster)
                evals_results.append(evals_result)
                val_pred.record(val_idx, booster.predict(cv_val_dmatrix))

                if len(bst) % kfold == 0:
                    logging.info("The metrics of round {} cross validation".format(int(len(bst) / kfold)))
                    print_cv_metric(num_round, evals_results[-kfold:])

            val_pred.save()

            if num_cv_round > 1:
                logging.info("The overall metrics of {}-round cross validation".format(num_cv_round))
                print_cv_metric(num_round, evals_results)

    except Exception as e:
        for customer_error_message in CUSTOMER_ERRORS:
            if customer_error_message in str(e):
                raise exc.UserError(str(e))

        exception_prefix = "XGB train call failed with exception"
        raise exc.AlgorithmError("{}:\n {}".format(exception_prefix, str(e)))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if is_master:
        if type(bst) is not list:
            model_location = os.path.join(model_dir, MODEL_NAME)
            bst.save_model(model_location)
            logging.debug("Stored trained model at {}".format(model_location))
        else:
            for fold in range(len(bst)):
                model_location = os.path.join(model_dir, f"{MODEL_NAME}-{fold}")
                bst[fold].save_model(model_location)
                logging.debug("Stored trained model {} at {}".format(fold, model_location))


def print_cv_metric(num_round, evals_results):
    cv_eval_report = f"[{num_round}]"
    for metric_name in evals_results[0]["train"]:
        for data_name in ["train", "validation"]:
            metric_val = [evals_result[data_name][metric_name][-1] for evals_result in evals_results]
            cv_eval_report += "\t{0}-{1}:{2:.5f}".format(data_name, metric_name, np.mean(metric_val))
    print(cv_eval_report)
