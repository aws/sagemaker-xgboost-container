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
from sagemaker_xgboost_container.data_utils import get_content_type, get_size_validated_dmatrix
from sagemaker_xgboost_container import distributed
from sagemaker_xgboost_container.algorithm_mode import channel_validation as cv
from sagemaker_xgboost_container.algorithm_mode import hyperparameter_validation as hpv
from sagemaker_xgboost_container.algorithm_mode import metrics as metrics_mod
from sagemaker_xgboost_container.algorithm_mode import train_utils
from sagemaker_xgboost_container.constants.xgb_constants import LOGISTIC_REGRESSION_LABEL_RANGE_ERROR, \
    MULTI_CLASS_LABEL_RANGE_ERROR, FEATURE_MISMATCH_ERROR, LABEL_PREDICTION_SIZE_MISMATCH, ONLY_POS_OR_NEG_SAMPLES, \
    BASE_SCORE_RANGE_ERROR, POISSON_REGRESSION_ERROR, TWEEDIE_REGRESSION_ERROR, REG_LAMBDA_ERROR


logger = logging.getLogger(__name__)


def sagemaker_train(train_config, data_config, train_path, val_path, model_dir, sm_hosts, sm_current_host):
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

    # Get Training and Validation Matrices
    file_type = get_content_type(validated_data_config)
    csv_weights = validated_train_config.get("csv_weights", 0)

    dtrain, dval = get_size_validated_dmatrix(train_path, val_path, file_type, csv_weights)

    train_args = dict(
        train_cfg=validated_train_config,
        dtrain=dtrain,
        dval=dval,
        model_dir=model_dir)

    # Obtain information about training resources to determine whether to set up Rabit or not
    num_hosts = len(sm_hosts)

    if num_hosts > 1:
        # Wait for hosts to find each other
        logging.info("Distributed node training with {} hosts: {}".format(num_hosts, sm_hosts))
        distributed.wait_hostname_resolution(sm_hosts)

        if not dtrain:
            logging.warning("Host {} does not have data. Will broadcast to cluster and will not be used in distributed"
                            " training.".format(sm_current_host))
        distributed.rabit_run(exec_fun=train_job, args=train_args, include_in_training=(dtrain is not None),
                              hosts=sm_hosts, current_host=sm_current_host, update_rabit_args=True)
    elif num_hosts == 1:
        if dtrain:
            logging.info("Single node training.")
            train_args.update({'is_master': True})
            train_job(**train_args)
        else:
            exc.UserError("No data in training channel path {}".format(train_path))
    else:
        raise exc.PlatformError("Number of hosts should be an int greater than or equal to 1")


def train_job(train_cfg, dtrain, dval, model_dir, is_master):
    """Train and save XGBoost model using data on current node.

    If doing distributed training, XGBoost will use rabit to sync the trained model between each boosting iteration.
    Trained model is only saved if 'is_master' is True.

    :param train_cfg: Training hyperparameter configurations
    :param dtrain: Path of training data
    :param dval: Path of validation data
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
    watchlist = [(dtrain, 'train'), (dval, 'validation')] if dval is not None else [(dtrain, 'train')]

    logging.info("Train matrix has {} rows".format(dtrain.num_row()))
    if dval:
        logging.info("Validation matrix has {} rows".format(dval.num_row()))

    try:
        logging.info(train_cfg)
        bst = xgb.train(train_cfg, dtrain, num_boost_round=num_round, evals=watchlist, feval=configured_feval,
                        early_stopping_rounds=early_stopping_rounds)
    except Exception as e:
        exception_prefix = "XGB train call failed with exception"
        if LOGISTIC_REGRESSION_LABEL_RANGE_ERROR in e.message:
            raise exc.UserError(
                "Label must be in [0,1] for logistic regression task. If input is in csv format, ensure the first "
                "column is the label.")
        elif MULTI_CLASS_LABEL_RANGE_ERROR in e.message:
            raise exc.UserError(
                "Label must be in [0, num_class) for multi classification task. If input is in csv format, "
                "ensure the first column is the label.")
        elif FEATURE_MISMATCH_ERROR in e.message:
            raise exc.UserError(
                "Feature names/Number of features mismatched between training and validation input. Please check "
                "features in training and validation data.")
        elif LABEL_PREDICTION_SIZE_MISMATCH in e.message:
            raise exc.UserError(
                "Given label size mismatched with prediction size. Please ensure the first column is label and the "
                "correct metric is applied.")
        elif ONLY_POS_OR_NEG_SAMPLES in e.message:
            raise exc.UserError(
                "Metric 'auc' cannot be computed with all positive or all negative samples. Please ensure labels in "
                "the datasets contain both positive and negative samples.")
        elif BASE_SCORE_RANGE_ERROR in e.message:
            raise exc.UserError("Base_score must be in (0,1) for logistic objective function.")
        elif POISSON_REGRESSION_ERROR in e.message:
            raise exc.UserError("For Poisson Regression, label must be non-negative")
        elif TWEEDIE_REGRESSION_ERROR in e.message:
            raise exc.UserError("For Tweedie Regression, label must be non-negative")
        elif REG_LAMBDA_ERROR in e.message:
            raise exc.UserError("Parameter reg_lambda should be greater equal to 0")
        else:
            raise exc.AlgorithmError("{}:\n {}".format(exception_prefix, e.message))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if is_master:
        model_location = model_dir + '/xgboost-model'
        pkl.dump(bst, open(model_location, 'wb'))
        logging.debug("Stored trained model at {}".format(model_location))
