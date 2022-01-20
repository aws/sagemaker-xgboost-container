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

import numpy as np
import xgboost as xgb

from scipy import stats
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
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
from sagemaker_xgboost_container.constants.sm_env_constants import SM_OUTPUT_DATA_DIR

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

    train_val_dmatrix = train_dmatrix
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
    combine_train_val = '_kfold' in validated_train_config
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
    # Parse arguments for train() API
    early_stopping_rounds = train_cfg.pop('early_stopping_rounds', None)
    num_round = train_cfg.pop("num_round")

    # Evaluation metrics to use with train() API
    tuning_objective_metric_param = train_cfg.pop("_tuning_objective_metric", None)
    eval_metric = train_cfg.get("eval_metric")
    cleaned_eval_metric, configured_feval = train_utils.get_eval_metrics_and_feval(
        tuning_objective_metric_param, eval_metric)
    if cleaned_eval_metric:
        train_cfg['eval_metric'] = cleaned_eval_metric
    else:
        train_cfg.pop('eval_metric', None)

    logging.info("Train matrix has {} rows and {} columns".format(train_dmatrix.num_row(), train_dmatrix.num_col()))
    if val_dmatrix:
        logging.info("Validation matrix has {} rows".format(val_dmatrix.num_row()))

    try:
        kfold = train_cfg.pop("_kfold", None)

        if kfold is None:
            xgb_model, iteration, callbacks, watchlist = get_callbacks_watchlist(
                train_cfg, train_dmatrix, val_dmatrix, model_dir, checkpoint_dir, is_master)
            add_debugging(callbacks=callbacks, hyperparameters=train_cfg, train_dmatrix=train_dmatrix,
                          val_dmatrix=val_dmatrix)

            bst = xgb.train(train_cfg, train_dmatrix, num_boost_round=num_round-iteration, evals=watchlist,
                            feval=configured_feval, early_stopping_rounds=early_stopping_rounds,
                            callbacks=callbacks, xgb_model=xgb_model, verbose_eval=False)

        else:
            num_cv_round = train_cfg.pop("_num_cv_round", 1)
            additional_output_path = os.environ[SM_OUTPUT_DATA_DIR]

            logging.info("Run {}-round of {}-fold cross validation with {} rows".format(num_cv_round,
                                                                                        kfold,
                                                                                        train_val_dmatrix.num_row()))

            bst = []
            evals_results = []

            num_class = train_cfg.get("num_class", None)
            objective = train_cfg.get("objective", None)
            # RepeatedStratifiedKFold expects X as array-like of shape (n_samples, n_features)
            classification_problem = num_class or objective.startswith("binary:")
            num_rows_in_dataset = train_val_dmatrix.num_row()
            X = range(num_rows_in_dataset)
            y = train_val_dmatrix.get_label() if classification_problem else None
            rkf = RepeatedStratifiedKFold(n_splits=kfold, n_repeats=num_cv_round) if y is not None \
                else RepeatedKFold(n_splits=kfold, n_repeats=num_cv_round)

            # For regression, ground truth and predicted values are stored. For classification, additionally
            # estimated probability for predicted class is stored. Predictions are aggregated across
            # different cross validation repeats - averaged for regression, mode is calculated for classification.
            data_to_store = np.zeros((num_rows_in_dataset, num_cv_round, 3 if classification_problem else 2))
            truth_idx = 0
            prob_idx = 1
            pred_idx = -1

            # This vector is needed to average over repeates of k-fold cross validaiton.
            cv_repeat_idx = np.zeros((num_rows_in_dataset,)).astype(int)

            for train_index, val_idx in rkf.split(X=X, y=y):
                cv_train_dmatrix = train_val_dmatrix.slice(train_index)
                cv_val_dmatrix = train_val_dmatrix.slice(val_idx)

                xgb_model, iteration, callbacks, watchlist = get_callbacks_watchlist(
                    train_cfg, cv_train_dmatrix, cv_val_dmatrix, model_dir, checkpoint_dir, is_master, len(bst))
                add_debugging(callbacks=callbacks, hyperparameters=train_cfg, train_dmatrix=cv_train_dmatrix,
                              val_dmatrix=cv_val_dmatrix)

                evals_result = {}
                logging.info("Train cross validation fold {}".format((len(bst) % kfold) + 1))
                booster = xgb.train(train_cfg, cv_train_dmatrix, num_boost_round=num_round-iteration,
                                    evals=watchlist, feval=configured_feval, evals_result=evals_result,
                                    early_stopping_rounds=early_stopping_rounds,
                                    callbacks=callbacks, xgb_model=xgb_model, verbose_eval=False)
                bst.append(booster)

                # store predictions on validation fold
                data_to_store[val_idx, cv_repeat_idx[val_idx], truth_idx] = cv_val_dmatrix.get_label()
                if classification_problem:
                    probabilities = booster.predict(cv_val_dmatrix)
                    if probabilities.ndim == 2:
                        # multi - class classification setting
                        pred_labels = probabilities.argmax(axis=-1)
                        probabilities = probabilities[np.arange(len(probabilities)), pred_labels]
                    else:
                        pred_labels = 1*(probabilities > 0.5)
                    data_to_store[val_idx, cv_repeat_idx[val_idx], prob_idx] = probabilities
                    data_to_store[val_idx, cv_repeat_idx[val_idx], pred_idx] = pred_labels
                else:
                    data_to_store[val_idx, cv_repeat_idx[val_idx], pred_idx] = booster.predict(cv_val_dmatrix)

                cv_repeat_idx[val_idx] += 1
                evals_results.append(evals_result)

                if len(bst) % kfold == 0:
                    logging.info("The metrics of round {} cross validation".format(int(len(bst) / kfold)))
                    print_cv_metric(num_round, evals_results[-kfold:])

            if classification_problem:
                proba_avg = data_to_store[:, :, prob_idx].mean(axis=1)
                # mode always return same dimension of output
                aggregated_data_to_store = stats.mode(data_to_store, axis=1).mode[:, 0, :]
                aggregated_data_to_store[:, prob_idx] = proba_avg
            else:
                aggregated_data_to_store = data_to_store.mean(axis=1)

            if not os.path.exists(additional_output_path):
                os.makedirs(additional_output_path)

            pred_path = os.path.join(additional_output_path, 'predictions.csv')
            logging.info(f"Storing predictions on cv folds averaged over all cv rounds in {pred_path} ")
            np.savetxt(pred_path, aggregated_data_to_store, delimiter=',', fmt='%f')

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
            with open(model_location, 'wb') as f:
                pkl.dump(bst, f, protocol=4)
            logging.debug("Stored trained model at {}".format(model_location))
        else:
            for fold in range(len(bst)):
                model_location = os.path.join(model_dir, f"{MODEL_NAME}-{fold}")
                with open(model_location, 'wb') as f:
                    pkl.dump(bst[fold], f, protocol=4)
                logging.debug("Stored trained model {} at {}".format(fold, model_location))


def get_callbacks_watchlist(train_cfg, train_dmatrix, val_dmatrix, model_dir, checkpoint_dir, is_master, fold=None):
    if checkpoint_dir and fold is not None:
        checkpoint_dir = os.path.join(checkpoint_dir, f"model-{fold}")

    # Set callbacks
    xgb_model, iteration = checkpointing.load_checkpoint(checkpoint_dir)
    if xgb_model is not None:
        if fold is not None:
            xgb_model = f"{xgb_model}-{fold}"
        logging.info("Checkpoint loaded from %s", xgb_model)
        logging.info("Resuming from iteration %s", iteration)

    callbacks = []
    callbacks.append(checkpointing.print_checkpointed_evaluation(start_iteration=iteration))
    if checkpoint_dir:
        save_checkpoint = checkpointing.save_checkpoint(checkpoint_dir, start_iteration=iteration)
        callbacks.append(save_checkpoint)

    # Parse arguments for intermediate model callback
    save_model_on_termination = train_cfg.pop('save_model_on_termination', "false")
    if save_model_on_termination == "true":
        model_name = f"{MODEL_NAME}-{fold}" if fold is not None else MODEL_NAME
        save_intermediate_model = checkpointing.save_intermediate_model(model_dir, model_name)
        callbacks.append(save_intermediate_model)
        add_sigterm_handler(model_dir, is_master)

    watchlist = [(train_dmatrix, 'train')]
    if val_dmatrix is not None:
        watchlist.append((val_dmatrix, 'validation'))

    return xgb_model, iteration, callbacks, watchlist


def print_cv_metric(num_round, evals_results):
    cv_eval_report = f"[{num_round}]"
    for metric_name in evals_results[0]['train']:
        for data_name in ["train", "validation"]:
            metric_val = [evals_result[data_name][metric_name][-1] for evals_result in evals_results]
            cv_eval_report += '\t{0}-{1}:{2:.5f}'.format(data_name, metric_name, np.mean(metric_val))
    print(cv_eval_report)
