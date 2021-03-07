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
import signal

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

MODEL_NAME = "xgboost-model"

logger = logging.getLogger(__name__)


def add_sigterm_handler(model_dir, is_master):
    """Stop training and cleanup model directory when SIGTERM is received.

    Model directory is only cleaned if is_master is True. Otherwise program terminates.

    :param model_dir: Directory where model is saved
    :param is_master: True if single node training, or the current node is the master node in distributed training
    """
    def _terminate():
        os._exit(0)

    def _cleanup_files(signo, frame):
        if is_master:
            train_utils.cleanup_dir(model_dir, MODEL_NAME)

        _terminate()

    signal.signal(signal.SIGTERM, _cleanup_files)


def get_validated_dmatrices(train_path, validate_path, content_type, csv_weights=0, is_pipe=False,
                            combine_train_val=False):
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
        logging.debug("File size need to be processed in the node: {}mb.".format(
            round((train_files_size + val_files_size) / (1024 * 1024), 2)))

        if train_files_size > 0:
            validate_data_file_path(train_path, content_type)
        if val_files_size > 0:
            validate_data_file_path(validate_path, content_type)

    train_dmatrix = get_dmatrix(train_path, content_type, csv_weights=csv_weights, is_pipe=is_pipe) \
        if train_files_size > 0 else None
    val_dmatrix = get_dmatrix(validate_path, content_type, csv_weights=csv_weights, is_pipe=is_pipe) \
        if val_files_size > 0 else None

    train_val_dmatrix = None
    if combine_train_val and train_dmatrix is not None and val_dmatrix is not None:
        logging.info("Read both train and validation data into one DMatrix")
        train_val_dmatrix = get_dmatrix([train_path, validate_path], content_type,
                                        csv_weights=csv_weights, is_pipe=is_pipe)

    return train_dmatrix, val_dmatrix, train_val_dmatrix


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
    combine_train_val = '_nfold' in validated_train_config
    train_dmatrix, val_dmatrix, train_val_dmatrix = get_validated_dmatrices(train_path, val_path, file_type,
                                                                            csv_weights, is_pipe, combine_train_val)

    checkpoint_dir = checkpoint_config.get("LocalPath", None)

    train_args = dict(
        train_cfg=validated_train_config,
        train_dmatrix=train_dmatrix,
        val_dmatrix=val_dmatrix,
        train_val_dmatrix=train_val_dmatrix,
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
    # Parse arguments for intermediate model callback
    save_model_on_termination = train_cfg.pop('save_model_on_termination', "false")

    # Parse arguments for train() API
    early_stopping_rounds = train_cfg.get('early_stopping_rounds')
    num_round = train_cfg.pop("num_round")

    # Evaluation metrics to use with train() API
    tuning_objective_metric_param = train_cfg.get("_tuning_objective_metric")
    eval_metric = train_cfg.get("eval_metric")
    cleaned_eval_metric, configured_feval, tuning_objective_metric = train_utils.get_eval_metrics_and_feval(
        tuning_objective_metric_param, eval_metric)
    if cleaned_eval_metric:
        train_cfg['eval_metric'] = cleaned_eval_metric
    else:
        train_cfg.pop('eval_metric', None)

    # Set callback evals
    watchlist = [(train_dmatrix, 'train')]
    if val_dmatrix is not None:
        watchlist.append((val_dmatrix, 'validation'))

    xgb_model, iterations = checkpointing.load_checkpoint(checkpoint_dir)
    num_round -= iterations
    if xgb_model is not None:
        logging.info("Checkpoint loaded from %s", xgb_model)
        logging.info("Resuming from iteration %s", iterations)

    callbacks = []
    callbacks.append(xgb.callback.EvaluationMonitor())
    if checkpoint_dir:
        callbacks.append(xgb.callback.TrainingCheckPoint(checkpoint_dir, iterations=iterations))

    if save_model_on_termination == "true":
        callbacks.append(checkpointing.SaveIntermediateModelCallBack(model_dir, MODEL_NAME))
        add_sigterm_handler(model_dir, is_master)

    if early_stopping_rounds:
        early_stopping_data_name = 'validation' if val_dmatrix else 'train'

        if tuning_objective_metric:
            early_stopping_metric_name = tuning_objective_metric[-1]
        elif eval_metric:
            early_stopping_metric_name = eval_metric[-1]

        if early_stopping_metric_name:
            maximize = train_utils.MAXIMIZE.get(early_stopping_metric_name, False)
            early_stop = xgb.callback.EarlyStopping(rounds=early_stopping_rounds,
                                                    data_name=early_stopping_data_name,
                                                    metric_name=early_stopping_metric_name,
                                                    maximize=maximize,
                                                    save_best=True)
            callbacks.append(early_stop)

    add_debugging(callbacks=callbacks, hyperparameters=train_cfg, train_dmatrix=train_dmatrix,
                  val_dmatrix=val_dmatrix)

    logging.info("Train matrix has {} rows and {} columns".format(train_dmatrix.num_row(), train_dmatrix.num_col()))
    if val_dmatrix:
        logging.info("Validation matrix has {} rows".format(val_dmatrix.num_row()))

    try:
        nfold = train_cfg.pop("_nfold", None)

        bst = xgb.train(train_cfg, train_dmatrix, num_boost_round=num_round, evals=watchlist, feval=configured_feval,
                        callbacks=callbacks, xgb_model=xgb_model, verbose_eval=False)

        if nfold is not None and train_val_dmatrix is not None:
            logging.info("Run {}-fold cross validation on the data of {} rows".format(nfold,
                                                                                      train_val_dmatrix.num_row()))
            # xgb.cv returns a pandas data frame of evaluation results.
            cv_eval_result = xgb.cv(train_cfg, train_val_dmatrix, nfold=nfold, num_boost_round=num_round,
                                    feval=configured_feval, verbose_eval=True, show_stdv=True, shuffle=False)

            logging.info("The final metrics of cross validation")
            cv_last_epoch = len(cv_eval_result.index) - 1
            cv_eval_report = f"[{cv_last_epoch}]"
            cv_eval_columns = cv_eval_result.columns
            # Skip the standard deviation columns
            for j in range(0, len(cv_eval_columns), 2):
                metric_name = cv_eval_columns[j][:-5].replace("test-", "validation-", 1)
                metric_val = cv_eval_result.at[cv_last_epoch, cv_eval_columns[j]]
                cv_eval_report += '\t{0}:{1:.5f}'.format(metric_name, metric_val)
            print(cv_eval_report)
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
        with open(model_location, 'wb') as f:
            pkl.dump(bst, f, protocol=4)
        logging.debug("Stored trained model at {}".format(model_location))
