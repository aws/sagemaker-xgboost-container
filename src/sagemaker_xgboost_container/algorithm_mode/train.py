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
import pickle as pkl
import os

import xgboost as xgb

from sagemaker_algorithm_toolkit import exceptions as exc
from sagemaker_algorithm_toolkit.channel_validation import Channel
from sagemaker_xgboost_container.data_utils import get_content_type, get_dmatrix, get_size, validate_data_file_path
from sagemaker_xgboost_container import distributed
from sagemaker_xgboost_container import checkpointing
from sagemaker_xgboost_container.algorithm_mode import channel_validation as cv
from sagemaker_xgboost_container.algorithm_mode import hyperparameter_validation as hpv
from sagemaker_xgboost_container.algorithm_mode import metrics as metrics_mod
from sagemaker_xgboost_container.algorithm_mode import train_utils
from sagemaker_xgboost_container.callback import add_debugging
from sagemaker_xgboost_container.constants.xgb_constants import CUSTOMER_ERRORS


logger = logging.getLogger(__name__)


def get_validated_dmatrices(train_path, validate_path, content_type, csv_weights=0, is_pipe=False):
    """Get training and validation Data Matrices for XGBoost training.

    Check size and format of both training and validation data channels, and return parsed
    Data Matrices.

    :param train_path:
    :param validate_path:
    :param content_type: Content type of data. Supports 'libsvm' or 'csv'
    :param csv_weights: 1 if instance weights are in the second column of csv data files; otherwise, 0
    :param is_pipe: Boolean to indicate if data is being read in pipe mode
    :return: Parsed xgb.DMatrix
    """
    train_files_size = get_size(train_path, is_pipe) if train_path else 0
    val_files_size = get_size(validate_path, is_pipe) if validate_path else 0

    if not is_pipe:
        logging.debug("File size need to be processed in the node: {}mb.".format(
            round((train_files_size + val_files_size) / (1024 * 1024), 2)))

        if train_files_size > 0:
            validate_data_file_path(train_path, content_type)
        if val_files_size > 0:
            validate_data_file_path(validate_path, content_type)

    train_dmatrix = get_dmatrix(train_path, content_type, csv_weights=csv_weights, is_pipe=is_pipe) \
        if train_files_size > 0 else None
    val_dmatrix = get_dmatrix(validate_path, content_type, is_pipe=is_pipe) \
        if val_files_size > 0 else None

    return train_dmatrix, val_dmatrix


def sagemaker_train(train_config, data_config, train_path, val_path, model_dir, sm_hosts, sm_current_host,
                    checkpoint_config):
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
    file_type = get_content_type(validated_data_config['train'].get("ContentType"))
    input_mode = validated_data_config['train'].get("TrainingInputMode")
    csv_weights = validated_train_config.get("csv_weights", 0)
    is_pipe = (input_mode == Channel.PIPE_MODE)

    validation_channel = validated_data_config.get('validation', None)
    train_dmatrix, val_dmatrix = get_validated_dmatrices(train_path, val_path, file_type, csv_weights, is_pipe)

    checkpoint_dir = checkpoint_config.get("LocalPath", None)

    train_args = dict(
        train_cfg=validated_train_config,
        train_dmatrix=train_dmatrix,
        val_dmatrix=val_dmatrix,
        model_dir=model_dir,
        checkpoint_dir=checkpoint_dir)

    # Obtain information about training resources to determine whether to set up Rabit or not
    num_hosts = len(sm_hosts)

    if num_hosts > 1:
        # Wait for hosts to find each other
        logging.info("Distributed node training with {} hosts: {}".format(num_hosts, sm_hosts))
        distributed.wait_hostname_resolution(sm_hosts)

        if not train_dmatrix:
            logging.warning("Host {} does not have data. Will broadcast to cluster and will not be used in distributed"
                            " training.".format(sm_current_host))
        distributed.rabit_run(exec_fun=train_job, args=train_args, include_in_training=(train_dmatrix is not None),
                              hosts=sm_hosts, current_host=sm_current_host, update_rabit_args=True)
    elif num_hosts == 1:
        if train_dmatrix:
            if validation_channel:
                if not val_dmatrix:
                    raise exc.UserError("No data in validation channel path {}".format(val_path))
            logging.info("Single node training.")
            train_args.update({'is_master': True})
            train_job(**train_args)
        else:
            raise exc.UserError("No data in training channel path {}".format(train_path))
    else:
        raise exc.PlatformError("Number of hosts should be an int greater than or equal to 1")


def train_job(train_cfg, train_dmatrix, val_dmatrix, model_dir, checkpoint_dir, is_master):
    """Train and save XGBoost model using data on current node.

    If doing distributed training, XGBoost will use rabit to sync the trained model between each boosting iteration.
    Trained model is only saved if 'is_master' is True.

    :param train_cfg: Training hyperparameter configurations
    :param train_dmatrix: Training Data Matrix
    :param val_dmatrix: Validation Data Matrix
    :param model_dir: Directory where model will be saved
    :param is_master: True if single node training, or the current node is the master node in distributed training.
    """
    # Parse arguments for train() API
    early_stopping_rounds = train_cfg.get('early_stopping_rounds')
    num_round = train_cfg["num_round"]

    # Evaluation metrics to use with train() API
    tuning_objective_metric_param = train_cfg.get("_tuning_objective_metric")
    eval_metric = train_cfg.get("eval_metric")
    cleaned_eval_metric, configured_feval = train_utils.get_eval_metrics_and_feval(
        tuning_objective_metric_param, eval_metric)
    if cleaned_eval_metric:
        train_cfg['eval_metric'] = cleaned_eval_metric
    else:
        train_cfg.pop('eval_metric', None)

    # Set callback evals
    watchlist = [(train_dmatrix, 'train')]
    if val_dmatrix is not None:
        watchlist.append((val_dmatrix, 'validation'))

    xgb_model, iteration = checkpointing.load_checkpoint(checkpoint_dir)
    num_round -= iteration
    if xgb_model is not None:
        logging.info("Checkpoint loaded from %s", xgb_model)
        logging.info("Resuming from iteration %s", iteration)

    callbacks = []
    callbacks.append(checkpointing.print_checkpointed_evaluation(start_iteration=iteration))
    if checkpoint_dir:
        save_checkpoint = checkpointing.save_checkpoint(checkpoint_dir, start_iteration=iteration)
        callbacks.append(save_checkpoint)

    add_debugging(callbacks=callbacks, hyperparameters=train_cfg, train_dmatrix=train_dmatrix,
                  val_dmatrix=val_dmatrix)

    logging.info("Train matrix has {} rows".format(train_dmatrix.num_row()))
    if val_dmatrix:
        logging.info("Validation matrix has {} rows".format(val_dmatrix.num_row()))

    try:
        bst = xgb.train(train_cfg, train_dmatrix, num_boost_round=num_round, evals=watchlist, feval=configured_feval,
                        early_stopping_rounds=early_stopping_rounds, callbacks=callbacks, xgb_model=xgb_model,
                        verbose_eval=False)
    except Exception as e:
        for customer_error_message in CUSTOMER_ERRORS:
            if customer_error_message in str(e):
                raise exc.UserError(str(e))

        exception_prefix = "XGB train call failed with exception"
        raise exc.AlgorithmError("{}:\n {}".format(exception_prefix, str(e)))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if is_master:
        model_location = model_dir + '/xgboost-model'
        pkl.dump(bst, open(model_location, 'wb'))
        logging.debug("Stored trained model at {}".format(model_location))
