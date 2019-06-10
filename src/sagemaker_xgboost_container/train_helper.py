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
from __future__ import unicode_literals

import csv
import logging
import os
import pickle as pkl
import psutil
import re
import socket
import subprocess
import sys
import time

import xgboost as xgb


# Copied from AI ALGS SDK
from sagemaker_xgboost_container.exceptions import AlgorithmError, CustomerValueError, PlatformError, CustomerError
# Commented out for time being
# from sg_algorithms_sdk.base.integration import setup_main_logger

# Copied from AI APPS XGB
from sagemaker_xgboost_container.bootstrap import file_prepare, cluster_config, start_daemons
from sagemaker_xgboost_container.constants.xgb_constants import LOGISTIC_REGRESSION_LABEL_RANGE_ERROR, \
    MULTI_CLASS_LABEL_RANGE_ERROR, FEATURE_MISMATCH_ERROR, LABEL_PREDICTION_SIZE_MISMATCH, ONLY_POS_OR_NEG_SAMPLES, \
    BASE_SCORE_RANGE_ERROR, POISSON_REGRESSION_ERROR, FNULL, TWEEDIE_REGRESSION_ERROR, REG_LAMBDA_ERROR
from sagemaker_xgboost_container.metrics.performance_metrics import create_runtime, get_runtime, \
    XGBoostPerformanceMetrics
from sagemaker_xgboost_container.metrics.custom_metrics import accuracy, f1  # noqa: F401


MODEL_DIR = os.getenv("ALGO_MODEL_DIR")
INPUT_DATA_PATH = os.getenv("ALGO_INPUT_DATA_DIR")
ALGORITHM_HOME_DIR = os.getenv("ALGORITHM_HOME_DIR")
HADOOP_PREFIX = os.environ['HADOOP_PREFIX']
OUTPUT_FAILED_FILE = os.getenv("ALGO_OUTPUT_FAILED_FILE")

EASE_MEMORY = 5120 * 1024 * 1024
YARN_MASTER_MEMORY = 1024 * 1024 * 1024
THRESHOLD_MEMORY = 1024 * 1024 * 1024
TRAIN_CHANNEL = 'train'
VAL_CHANNEL = 'validation'
HPO_SEPARATOR = ':'


#######################################################################
#    Helper functions for standalone training
########################################################################

# parameters from xgboost
def get_int(param_value, param_name):
    try:
        result = int(param_value)
    except ValueError:
        raise CustomerValueError("Parameter {} expects integer input.".format(param_name))
    return result


def get_float(param_value, param_name):
    try:
        result = float(param_value)
    except ValueError:
        raise CustomerValueError("Parameter {} expects float input.".format(param_name))
    return result


def get_params(cfg):
    param = {}
    # param for general
    if "booster" in cfg:  # Default: 'gbtree'
        if cfg["booster"] not in ["gbtree", "gblinear", "dart"]:
            raise CustomerValueError(
                "Parameter 'booster' should be one of these options: 'gbtree', 'gblinear', 'dart'.")
        if cfg.get('booster') == 'gblinear':
            if "updater" not in cfg:
                raise CustomerValueError("Linear updater should be one of these options: 'shotgun', 'coor_descent'.")
        param["booster"] = cfg["booster"]
    if "silent" in cfg:  # Default: 0
        tmp = get_int(cfg["silent"], "silent")
        if tmp not in [0, 1]:
            raise CustomerValueError("Parameter 'silent' should be either '0' or '1'.")
        param["silent"] = tmp
    if "nthread" in cfg:  # Default: maximum allowed threads
        tmp = get_int(cfg["nthread"], "nthread")
        if tmp < 1:
            raise CustomerValueError("Parameter 'nthread' should be greater or equal to 1.")
        param["nthread"] = tmp
    # params for Tree Boosters
    if "eta" in cfg:  # Default: 0.3
        tmp = get_float(cfg["eta"], "eta")
        if tmp > 1 or tmp < 0:
            raise CustomerValueError("Parameter 'eta' should be in range [0,1].")
        param["eta"] = tmp
    if "gamma" in cfg:  # Default: 0
        tmp = get_float(cfg["gamma"], "gamma")
        if tmp < 0:
            raise CustomerValueError("Parameter 'gamma' should be greater than or equal to 0.")
        param["gamma"] = tmp
    if "max_depth" in cfg:  # Default: 6
        tmp = get_int(cfg["max_depth"], "max_depth")
        if tmp < 0:
            raise CustomerValueError("Parameter 'max_depth' should be greater than or equal to 0.")
        param["max_depth"] = tmp
    if "min_child_weight" in cfg:  # Default: 1
        tmp = get_float(cfg["min_child_weight"], "min_child_weight")
        if tmp < 0:
            raise CustomerValueError("Parameter 'min_child_weight' should be greater than or equal to 0.")
        param["min_child_weight"] = tmp
    if "max_delta_step" in cfg:  # Default: 0
        tmp = get_float(cfg["max_delta_step"], "max_delta_step")
        if tmp < 0:
            raise CustomerValueError("Parameter 'max_delta_step' should be greater than or equal to 0.")
        param["max_delta_step"] = tmp
    if "subsample" in cfg:  # Default: 1
        tmp = get_float(cfg["subsample"], "subsample")
        if tmp <= 0 or tmp > 1:
            raise CustomerValueError("Parameter 'subsample' should be in range (0,1].")
        param["subsample"] = tmp
    if "colsample_bytree" in cfg:  # Default: 1
        tmp = get_float(cfg["colsample_bytree"], "colsample_bytree")
        if tmp <= 0 or tmp > 1:
            raise CustomerValueError("Parameter 'colsample_bytree' should be in range (0,1].")
        param["colsample_bytree"] = tmp
    if "colsample_bylevel" in cfg:  # Default: 1
        tmp = get_float(cfg["colsample_bylevel"], "colsample_bylevel")
        if tmp <= 0 or tmp > 1:
            raise CustomerValueError("Parameter 'colsample_bylevel' should be in range (0,1].")
        param["colsample_bylevel"] = tmp
    if "lambda" in cfg:  # Default: 1
        param["lambda"] = get_float(cfg["lambda"], "lambda")
    if "alpha" in cfg:  # Default: 0
        param["alpha"] = get_float(cfg["alpha"], "alpha")
    if "tree_method" in cfg:  # Default: 'auto'
        if "gpu" in cfg["tree_method"]:
            raise CustomerValueError("GPU training is not supported yet.")
        if cfg["tree_method"] not in ["auto", "exact", "approx", "hist"]:
            raise CustomerValueError(
                "Parameter 'tree_method' should be one of these options: 'auto', 'exact', 'approx', 'hist'.")
        param["tree_method"] = cfg["tree_method"]
    if "sketch_eps" in cfg:  # Default: 0.03
        tmp = get_float(cfg["sketch_eps"], "sketch_eps")
        if tmp <= 0 or tmp >= 1:
            raise CustomerValueError("Parameter 'sketch_eps' should be in range (0,1).")
        param["sketch_eps"] = tmp
    if "scale_pos_weight" in cfg:  # Default: 1
        param["scale_pos_weight"] = get_float(cfg["scale_pos_weight"], "scale_pos_weight")
    if "updater" in cfg:  # Default: 'grow_colmaker,prune'
        if '[' in cfg["updater"] or ']' in cfg["updater"]:
            raise CustomerValueError("Do not expect bracket for input value of 'updater'. Use format as 'A,B,C...'.")
        try:
            tmp_list = cfg["updater"].split(',')
        except Exception:
            raise CustomerValueError(
                "Input value of 'updater' should be concatenated by comma. Use format as 'A,B,C...'.")

        valid_tree_plugins = ['grow_colmaker', 'distcol', 'grow_histmaker', 'grow_local_histmaker',
                              'grow_skmaker', 'sync', 'refresh', 'prune']
        valid_tree_build_plugins = ['grow_colmaker', 'distcol', 'grow_histmaker',
                                    'grow_local_histmaker', 'grow_colmaker']
        valid_linear_plugins = ['shotgun', 'coord_descent']
        valid_process_update_plugins = ['refresh', 'prune']
        if cfg.get('booster') == 'gblinear':
            # validate only one linear updater is selected
            if not (len(tmp_list) == 1 and tmp_list[0] in valid_linear_plugins):
                raise CustomerValueError("Linear updater should be one of these options: 'shotgun', 'coor_descent'.")
        elif cfg.get('process_type') == 'update':
            if not all(x in valid_process_update_plugins for x in tmp_list):
                raise CustomerValueError("process_type 'update' can only be used with updater 'refresh' and 'prune'")
        else:
            if not all(x in valid_tree_plugins for x in tmp_list):
                raise CustomerValueError(
                    "Tree updater should be selected from these options: 'grow_colmaker', 'distcol', 'grow_histmaker', "
                    "'grow_local_histmaker', 'grow_skmaker', 'sync', 'refresh', 'prune', 'shortgun', 'coord_descent'.")
            # validate only one tree updater is selected
            counter = 0
            for tmp in tmp_list:
                if tmp in valid_tree_build_plugins:
                    counter += 1
            if counter > 1:
                raise CustomerValueError("Only one tree grow plugin can be selected. Choose one from the"
                                         "following: 'grow_colmaker', 'distcol', 'grow_histmaker', "
                                         "'grow_local_histmaker', 'grow_skmaker'")
        param["updater"] = cfg["updater"]
    if "dsplit" in cfg:  # Default: row
        if cfg["dsplit"] not in ["row", "col"]:
            raise CustomerValueError("Parameter 'dsplit' should be one of these options: 'row', 'col'.")
        param["dsplit"] = cfg["dsplit"]
    if "prob_buffer_row" in cfg:  # Default: 1
        param["prob_buffer_row"] = get_float(cfg["prob_buffer_row"], "prob_buffer_row")
    if "refresh_leaf" in cfg:  # Default: 1
        tmp = get_int(cfg["refresh_leaf"], "refresh_leaf")
        if tmp not in [0, 1]:
            raise CustomerValueError("Parameter 'refresh_leaf' should be either '0' or '1'.")
        param["refresh_leaf"] = tmp
    if "process_type" in cfg:  # Default: 'default'
        if cfg["process_type"] not in ["default", "update"]:
            raise CustomerValueError("Parameter 'process_type' should be one of these options: 'default', 'update'.")
        if cfg.get('process_type') == 'update':
            if "updater" not in cfg:
                raise CustomerValueError("process_type 'update' can only be used with updater 'refresh' and 'prune'")
        param["process_type"] = cfg["process_type"]
    if "grow_policy" in cfg:  # Default: 'depthwise'
        if cfg["grow_policy"] not in ["depthwise", "lossguide"]:
            raise CustomerValueError(
                "Parameter 'grow_policy' should be one of these options: 'depthwise', 'lossguide'.")
        param["grow_policy"] = cfg["grow_policy"]
    if "max_leaves" in cfg:  # Default: 0
        param["max_leaves"] = get_int(cfg["max_leaves"], "max_leaves")
    if "max_bins" in cfg:  # Default: 256
        param["max_bins"] = get_int(cfg["max_bins"], "max_bins")
    if "predictor" in cfg:  # Default: cpu_predictor
        if "gpu" in cfg["predictor"]:
            raise CustomerValueError("GPU training is not supported yet.")
        if cfg["predictor"] not in ["cpu_predictor", "gpu_predictor"]:
            raise CustomerValueError("Parameter 'predictor' should be one of these options: 'cpu_predictor'.")
        param["predictor"] = cfg["predictor"]
    # params specific for DART Booster
    if "sample_type" in cfg:  # Default: 'uniform'
        if cfg["sample_type"] not in ["uniform", "weighted"]:
            raise CustomerValueError("Parameter 'sample_type' should be one of these options: 'uniform', 'weighted'.")
        param["sample_type"] = cfg["sample_type"]
    if "normalize_type" in cfg:  # Default: 'tree'
        if cfg["normalize_type"] not in ["tree", "forest"]:
            raise CustomerValueError("Parameter 'normalize_type' should be one of these options: 'tree', 'forest'.")
        param["normalize_type"] = cfg["normalize_type"]
    if "rate_drop" in cfg:  # Default: 0.0
        tmp = get_float(cfg["rate_drop"], "rate_drop")
        if tmp < 0 or tmp > 1:
            raise CustomerValueError("Parameter 'rate_drop' should be in range [0,1].")
        param["rate_drop"] = tmp
    if "one_drop" in cfg:  # Default: 0
        tmp = get_int(cfg["one_drop"], "one_drop")
        if tmp not in [0, 1]:
            raise CustomerValueError("Parameter 'one_drop' should be either 0 or 1.")
        param["one_drop"] = tmp
    if "skip_drop" in cfg:  # Default: 0.0
        tmp = get_float(cfg["skip_drop"], "skip_drop")
        if tmp < 0 or tmp > 1:
            raise CustomerValueError("Parameter 'skip_drop' should be in range [0,1].")
        param["skip_drop"] = tmp
    # params specific for Linear Booster
    if "lambda_bias" in cfg:  # Default: 0.0
        param["lambda_bias"] = get_float(cfg["lambda_bias"], "lambda_bias")
    # params specific for Tweedie Regression
    if "tweedie_variance_power" in cfg:  # Default: 1.5
        tmp = get_float(cfg["tweedie_variance_power"], "tweedie_variance_power")
        if tmp <= 1 or tmp >= 2:
            raise CustomerValueError("Parameter 'tweedie_variance_power' should be in range (1,2).")
        param["tweedie_variance_power"] = tmp
    # params for Learning Task
    if "objective" in cfg:  # Default: 'reg:linear'
        if cfg["objective"] not in ["reg:linear", "reg:logistic", "binary:logistic", "binary:logitraw", "count:poisson",
                                    "multi:softmax", "multi:softprob", "rank:pairwise", "reg:gamma", "reg:tweedie"]:
            raise CustomerValueError(
                "Parameter 'objective' should be one of these options: 'reg:linear', 'reg:logistic', "
                "'binary:logistic', 'binary:logitraw', 'count:poisson', 'multi:softmax', 'multi:softprob', "
                "'rank:pairwise', 'reg:gamma', 'reg:tweedie'.")
        if cfg["objective"] == "multi:softmax" or cfg["objective"] == "multi:softprob":
            if "num_class" not in cfg:
                raise CustomerValueError("Require input for parameter 'num_class' for multi classification.")
        param["objective"] = cfg["objective"]
    if "num_class" in cfg:  # Default: N/A
        if "objective" not in cfg or cfg["objective"] not in ["multi:softmax", "multi:softprob"]:
            raise CustomerValueError(
                "Do not need to setup parameter 'num_class' for learning task other than multi classification.")
        tmp = get_int(cfg["num_class"], "num_class")
        if tmp < 2:
            raise CustomerValueError("Parameter 'num_class' should be greater or equal to 2.")
        param["num_class"] = tmp
    if "base_score" in cfg:  # Default: 0.5
        param["base_score"] = get_float(cfg["base_score"], "base_score")
    if "_tuning_objective_metric" in cfg:
        """
        TODO: validation to be added
        """
        param["_tuning_objective_metric"] = cfg["_tuning_objective_metric"]
    if "eval_metric" in cfg:  # Default: according to objective
        if '[' in cfg["eval_metric"] or ']' in cfg["eval_metric"]:
            raise CustomerValueError(
                "Do not expect bracket for input value of 'eval_metric'. Use format as 'A,B,C...'.")
        try:
            tmp_list = cfg["eval_metric"].split(',')
        except Exception:
            raise CustomerValueError(
                "Input value of 'eval_metric' should be concatenated by comma. Use format as 'A,B,C...'.")
        supported_metric = ["rmse", "mae", "logloss", "error", "merror", "mlogloss", "auc", "ndcg", "map",
                            "poisson-nloglik", "gamma-nloglik", "gamma-deviance", "tweedie-nloglik"]
        for metric in tmp_list:
            if metric == 'auc':
                if "objective" not in cfg or cfg["objective"] not in \
                        ["binary:logistic", "binary:logitraw", "multi:softmax", "multi:softprob",
                         "reg:logistic", "rank:pairwise", "binary:hinge"]:
                    raise CustomerValueError("Metric 'auc' can only be applied for classification and ranking problem.")
            if "<function" in metric:
                raise CustomerValueError("User defined evaluation metric {} is not supported yet.".format(metric))
            if "@" not in metric and metric not in supported_metric:
                raise CustomerValueError(
                    "Metric '{}' is not supported. Parameter 'eval_metric' should be one of these options:"
                    "'rmse', 'mae', 'logloss', 'error', 'merror', 'mlogloss', 'auc', 'ndcg', 'map', "
                    "'poisson-nloglik', 'gamma-nloglik', 'gamma-deviance', 'tweedie-nloglik'.".format(metric))
            if "@" in metric:
                metric_name = metric.split('@')[0].strip()
                metric_threshold = metric.split('@')[1].strip()
                if metric_name not in ["error", "ndcg", "map"]:
                    raise CustomerValueError(
                        "Metric '{}' is not supported. Parameter 'eval_metric' with customized threshold should "
                        "be one of these options: 'error', 'ndcg', 'map'.".format(metric))
                try:
                    tmp = float(metric_threshold)
                except ValueError:
                    raise CustomerValueError("Threshold value 't' in '{}@t' expects float input.".format(metric_name))
        param["eval_metric"] = tmp_list
    if "seed" in cfg:
        param["seed"] = get_int(cfg["seed"], "seed")

    return param


def _valid_file(file_path, file_name):
    """Return if the file is a valid data file.
    :param file_path: str
    :param file_name: str
    :return: bool
    """
    if not os.path.isfile(os.path.join(file_path, file_name)):
        return False
    if file_name.startswith('.') or file_name.startswith('_'):
        return False
    # avoid XGB cache file
    if '.cache' in file_name:
        if 'dtrain' in file_name or 'dval' in file_name:
            return False
    return True


def validate_file_format(dir_path, file_type):
    if not os.path.exists(dir_path):
        return
    else:
        files_path = None
        for root, dirs, files in os.walk(dir_path):
            if dirs == []:
                files_path = root
                break
        data_files = [f for f in os.listdir(files_path) if _valid_file(files_path, f)]
        if file_type.lower() == 'csv':
            for data_file in data_files:
                validate_csv_format(os.path.join(files_path, data_file))
        elif file_type.lower() == 'libsvm':
            for data_file in data_files:
                validate_libsvm_format(os.path.join(files_path, data_file))


def validate_csv_format(file_path):
    with open(file_path, 'r', errors='ignore') as f:
        first_line = f.readline()
        # validate it's not libsvm
        if ' ' in first_line and ':' in first_line:
            raise CustomerError(
                "Blankspace and colon found in firstline '{}...' of file '{}'. Please ensure the file is in csv "
                "format.".format(
                    first_line[:50], file_path.split('/')[-1]))
        # validate no header line
        match_object = re.search('[a-df-zA-DF-Z]', first_line)
        if match_object is not None:
            raise CustomerError("Non-numeric value '{}' found in the header line '{}...' of file '{}'. "
                                "CSV format require no header line in it. If header line is already removed, "
                                "XGBoost does not accept non-numeric value in the data.".format(
                                    match_object.group(0), first_line[:50], file_path.split('/')[-1]))


def validate_libsvm_format(file_path):
    with open(file_path, 'r', errors='ignore') as f:
        first_line = f.readline()
        # validate it's not csv
        if not (' ' in first_line and ':' in first_line):
            raise CustomerError("Blankspace and colon not found in firstline '{}...' of file '{}'. ContentType by "
                                "defaullt is in libsvm. Please ensure the file is in libsvm format.".format(
                                    first_line[:50], file_path.split('/')[-1]))


def get_csv_dmatrix(files_path, csv_weights):
    # infer delimiter of CSV input
    csv_file = [f for f in os.listdir(files_path) if os.path.isfile(os.path.join(files_path, f))][0]
    with open(os.path.join(files_path, csv_file), errors='ignore') as f:
        sample_text = f.readline().strip()[:512]
    try:
        delimiter = csv.Sniffer().sniff(sample_text).delimiter
        logging.info("Determined delimiter of CSV input is \'{}\'".format(delimiter))
    except Exception as e:
        raise CustomerError("Could not determine delimiter on line {}:\n{}".format(sample_text[:50], e))

    try:
        if csv_weights == 1:
            dmatrix = xgb.DMatrix(
                '{}?format=csv&label_column=0&delimiter={}&weight_column=1'.format(files_path, delimiter))
        else:
            dmatrix = xgb.DMatrix('{}?format=csv&label_column=0&delimiter={}'.format(files_path, delimiter))
    except Exception as e:
        raise CustomerError("Failed to load csv data with exception:\n{}".format(e))
    return dmatrix


def get_libsvm_dmatrix(files_path, exceed_memory):
    if exceed_memory:
        logging.info("Insufficient memory available in the instance. Using external memory for libsvm input.\
                      Switch to larger instance with more RAM/distributed mode for better performance and lower cost.")
        if 'train' in files_path:
            files_path = files_path + '#' + files_path + '/dtrain.cache'
        else:
            files_path = files_path + '#' + files_path + '/dval.cache'
    try:
        dmatrix = xgb.DMatrix(files_path)
    except Exception as e:
        raise CustomerError("Failed to load libsvm data with exception:\n{}".format(e))
    return dmatrix


def get_dmatrix(dir_path, file_type, exceed_memory, csv_weights=0):
    if not os.path.exists(dir_path):
        return None
    else:
        files_path = None
        for root, dirs, files in os.walk(dir_path):
            if dirs == []:
                files_path = root
                break
        if file_type.lower() == 'csv':
            if exceed_memory:
                raise CustomerError("Insufficient memory available in the instance. External memory for csv input not "
                                    "supported.\
                                 Switch to larger instance with more RAM/distributed mode for training.")
            dmatrix = get_csv_dmatrix(files_path, csv_weights)
        elif file_type.lower() == 'libsvm':
            dmatrix = get_libsvm_dmatrix(files_path, exceed_memory)

        if dmatrix.get_label().size == 0:
            raise CustomerError(
                "Got input data without labels. Please check the input data set. "
                "If training job is running on multiple instances, please switch "
                "to using single instance if number of records in the data set "
                "is less than number of workers (16 * number of instance) in the cluster.")

    return dmatrix


def get_size(dir_path):
    if not os.path.exists(dir_path):
        logging.info('Path {} does not exist!'.format(dir_path))
        return 0
    else:
        total_size = 0
        for root, dirs, files in os.walk(dir_path):
            for f in files:
                if f.startswith('.'):
                    raise CustomerError("Hidden file found in the data path! Remove that before training.")
                fp = os.path.join(root, f)
                total_size += os.path.getsize(fp)
        if total_size == 0:
            raise CustomerError(
                "No file found in path {}. If you are using ShardedByS3Key, make sure every machine get at least one "
                "partition file".format(
                    dir_path))
        return total_size


def validate_input_mode(data_cfg):
    input_mode = data_cfg[TRAIN_CHANNEL].get("TrainingInputMode")
    if input_mode.lower() == 'pipe':
        raise CustomerError("Pipe input mode for SageMaker XGBoost is not supported.")


def get_content_type(data_cfg):
    tmp = data_cfg[TRAIN_CHANNEL].get("ContentType")
    file_type = None
    if tmp is None or tmp.lower() == 'libsvm' or tmp.lower() == 'text/libsvm':
        file_type = 'libsvm'
    elif tmp.lower() == 'csv' or tmp.lower() == 'text/csv':
        file_type = 'csv'
    else:
        raise CustomerError("ContentType should be one of these options: 'text/libsvm', 'text/csv'.")
    return file_type


def validate_channel_name(channels):
    if TRAIN_CHANNEL not in channels:
        raise CustomerError("Channelname 'train' is required for training")
    if len(channels) == 2 and VAL_CHANNEL not in channels:
        raise CustomerError("Channelname 'validation' is used for validation input")


def get_union_metrics(metric_a, metric_b):
    """
    Return metric list after union metric_a and metric_b
    :param metric_a: list
    :param metric_b: list
    :return: union metrics list from metric a and b
    """
    if metric_a is None and metric_b is None:
        return None
    elif metric_a is None:
        return metric_b
    elif metric_b is None:
        return metric_a
    else:
        metric_list = list(set(metric_a).union(metric_b))
        return metric_list


def train_job(resource_config, train_cfg, data_cfg):
    param = get_params(train_cfg)
    if train_cfg.get("num_round") is None:
        raise CustomerValueError("Require input for parameter num_round")

    early_stopping_rounds = get_int(train_cfg.get("early_stopping_rounds"), "early_stopping_rounds") if train_cfg.get(
        "early_stopping_rounds") is not None else None
    num_round = get_int(train_cfg.get("num_round"), "num_round")
    csv_weights = get_int(train_cfg.get("csv_weights"), "csv_weights") if train_cfg.get(
        "csv_weights") is not None else 0
    if csv_weights not in [0, 1]:
        raise CustomerValueError("Parameter 'csv_weights' should be either '0' or '1'.")

    # performance metrics for hpo
    create_runtime(param, data_cfg)
    get_runtime()

    # union 'eval_metric' with '_tuning_objective_metric' to support HPO
    tuning_objective_metric = None
    custom_objective_metric = None
    if param.get("_tuning_objective_metric") is not None:
        tuning_objective_metric_tuple = XGBoostPerformanceMetrics.decode_metric_name(
            param.get("_tuning_objective_metric"))
        tuning_objective_metric_name = tuning_objective_metric_tuple.metric_name
        tuning_objective_metric = tuning_objective_metric_name.split(',')
        logging.info('Setting up HPO optimized metric to be : {}'.format(tuning_objective_metric_name))

        if tuning_objective_metric_name in ["f1", "accuracy"]:
            custom_objective_metric = globals()[tuning_objective_metric_name]

    eval_metric = param.get("eval_metric")
    if custom_objective_metric and eval_metric:
        raise CustomerValueError("Evaluation metrics 'accuracy' and 'f1' cannot be used with other metrics but"
                                 "'eval_metric' set to {}".format(eval_metric))
    elif not custom_objective_metric:
        union_metrics = get_union_metrics(tuning_objective_metric, eval_metric)
        if union_metrics is not None:
            param["eval_metric"] = union_metrics

    num_hosts = len(resource_config["hosts"])
    channels = list(data_cfg.keys())
    validate_channel_name(channels)
    validate_input_mode(data_cfg)
    # Set default content type as libsvm
    file_type = get_content_type(data_cfg)

    train_path = INPUT_DATA_PATH + '/' + TRAIN_CHANNEL
    val_path = INPUT_DATA_PATH + '/' + VAL_CHANNEL

    s3_dist_type_train = data_cfg[TRAIN_CHANNEL].get("S3DistributionType")
    s3_dist_type_val = data_cfg[VAL_CHANNEL].get("S3DistributionType") if data_cfg.get(
        VAL_CHANNEL) is not None else None

    train_files_size = get_size(train_path)
    val_files_size = get_size(val_path)
    real_train_mem_size = train_files_size / num_hosts if s3_dist_type_train == "FullyReplicated" else train_files_size
    real_val_mem_size = val_files_size / num_hosts if s3_dist_type_val == "FullyReplicated" else val_files_size

    # Keep 1GB memory as thredshold to avoid drain out all the memory
    mem_size = psutil.virtual_memory().available
    real_mem_size = mem_size - EASE_MEMORY - THRESHOLD_MEMORY

    exceed_memory = (real_train_mem_size + real_val_mem_size) > real_mem_size
    logging.info("File size need to be processed in the node: {}. Available memory size in the node: {}".format(
        str(round((real_train_mem_size + real_val_mem_size) / (1024 * 1024), 2)) + 'mb',
        str(round(real_mem_size / (1024 * 1024), 2)) + 'mb'))

    # remove redundant format checking for distributed XGB
    if num_hosts == 1:
        validate_file_format(train_path, file_type)
        validate_file_format(val_path, file_type)

    dtrain = get_dmatrix(train_path, file_type, exceed_memory, csv_weights=csv_weights)
    dval = get_dmatrix(val_path, file_type, exceed_memory)
    watchlist = [(dtrain, 'train'), (dval, 'validation')] if dval is not None else [(dtrain, 'train')]

    try:
        bst = xgb.train(param, dtrain, num_boost_round=num_round, evals=watchlist, feval=custom_objective_metric,
                        early_stopping_rounds=early_stopping_rounds)
    except Exception as e:
        exception_prefix = "XGB train call failed with exception"
        if LOGISTIC_REGRESSION_LABEL_RANGE_ERROR in e.message:
            raise CustomerError(
                "Label must be in [0,1] for logistic regression task. If input is in csv format, ensure the first "
                "column is the label.")
        elif MULTI_CLASS_LABEL_RANGE_ERROR in e.message:
            raise CustomerError(
                "Label must be in [0, num_class) for multi classification task. If input is in csv format, "
                "ensure the first column is the label.")
        elif FEATURE_MISMATCH_ERROR in e.message:
            raise CustomerError(
                "Feature names/Number of features mismatched between training and validation input. Please check "
                "features in training and validation data.")
        elif LABEL_PREDICTION_SIZE_MISMATCH in e.message:
            raise CustomerError(
                "Given label size mismatched with prediction size. Please ensure the first column is label and the "
                "correct metric is applied.")
        elif ONLY_POS_OR_NEG_SAMPLES in e.message:
            raise CustomerError(
                "Metric 'auc' cannot be computed with all positive or all negative samples. Please ensure labels in "
                "the datasets contain both positive and negative samples.")
        elif BASE_SCORE_RANGE_ERROR in e.message:
            raise CustomerError("Base_score must be in (0,1) for logistic objective function.")
        elif POISSON_REGRESSION_ERROR in e.message:
            raise CustomerError("For Poisson Regression, label must be non-negative")
        elif TWEEDIE_REGRESSION_ERROR in e.message:
            raise CustomerError("For Tweedie Regression, label must be non-negative")
        elif REG_LAMBDA_ERROR in e.message:
            raise CustomerError("Parameter reg_lambda should be greater equal to 0")
        else:
            raise AlgorithmError("{}:\n {}".format(exception_prefix, e.message))

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    pkl.dump(bst, open(MODEL_DIR + '/xgboost-model', 'wb'))


#######################################################################
#    Helper functions for distributed training
########################################################################

def get_yarn_config(train_cfg, num_hosts):
    # Adding yarn parameters configurations
    # Get paramters from user if given
    num_workers = train_cfg.get("num_workers", None)
    cores = train_cfg.get("worker_cores", None)
    mem = train_cfg.get("worker_memory", None)

    # Auto config for yarn parameters if not given
    mem_size = psutil.virtual_memory().total / (1024 * 1024 * 1024)
    cores_count = psutil.cpu_count(logical=False)
    logging.info("Memory/core ratio is {}".format(round(mem_size / cores_count, 2)))
    mem_bound = mem_size / cores_count <= 4

    if num_workers is None:
        if mem_bound:
            if num_hosts <= 5:
                num_workers = num_hosts * 16
            elif 5 < num_hosts <= 10:
                num_workers = num_hosts * 8
            elif 10 < num_hosts <= 15:
                num_workers = num_hosts * 4
            elif 15 < num_hosts <= 60:
                num_workers = num_hosts * 2
            else:
                num_workers = num_hosts
        else:
            num_workers = num_hosts
    else:
        num_workers = int(num_workers)

    workers_per_host = num_workers / num_hosts
    cores = cores_count if cores is None else cores  # each worker use up all the cores to gurantee cpu utilization
    mem = str(round((psutil.virtual_memory().total / (1024 * 1024 * 1024)) / workers_per_host,
                    2)) + 'g' if mem is None else mem  # divide memory among workers in a node
    return num_workers, cores, mem


def get_app_status():
    app_list = []
    yarn_app_logs_path = os.path.join(HADOOP_PREFIX, 'logs/userlogs')
    while not os.path.exists(yarn_app_logs_path):
        time.sleep(1)
    while app_list == []:
        time.sleep(1)
        app_list = os.listdir(yarn_app_logs_path)
    app_id = max(app_list)
    yarn_cmd = '{}/bin/yarn application -status {}'.format(HADOOP_PREFIX, app_id)
    yarn_status, yarn_final_status = 'RUNNING', None
    while yarn_status == 'RUNNING':
        time.sleep(60)
        yarn_status_report = subprocess.check_output(yarn_cmd, shell=True, stderr=FNULL).split('\n')
        yarn_status_lines = [line for line in yarn_status_report if 'State' in line]
        yarn_status = yarn_status_lines[0].split(':')[1].strip()
        yarn_final_status = yarn_status_lines[1].split(':')[1].strip()
    return yarn_status, yarn_final_status


def submit_yarn_job(train_cfg, host_ip, num_hosts):
    # populate workers to expose certain amount of parallelism
    num_workers, cores, mem = get_yarn_config(train_cfg, num_hosts)
    args = '--cluster=yarn --num-workers={} --host-ip={} --worker-cores={} --worker-memory={}'.format(num_workers,
                                                                                                      host_ip, cores,
                                                                                                      mem)

    logging.info("Yarn setup: number of workers: {}, "
                 "physical cores per worker: {}, "
                 "physical memory per worker: {}.".format(num_workers, cores, mem))

    python_path = 'PYTHONPATH=/xgboost/python-package'

    cmd_submit = "{} /xgboost/dmlc-core/tracker/dmlc-submit \
                {} python {}/yarnjob.py".format(python_path, args, ALGORITHM_HOME_DIR)

    subprocess.Popen(cmd_submit, shell=True, stdout=FNULL)
    logging.info("Yarn job submitted successfully.")
    yarn_status, yarn_final_status = get_app_status()

    if not (yarn_status == 'FINISHED' and yarn_final_status == 'SUCCEEDED'):
        logging.error("Yarn tasks failed! Report logs from worker.")
        try:
            yarn_app_logs_path = os.path.join(HADOOP_PREFIX, 'logs/userlogs')
            app_id = os.listdir(yarn_app_logs_path)[-1]
            container_ids = os.listdir(os.path.join(yarn_app_logs_path, app_id))
            for container_id in container_ids:
                log_file_path = os.path.join(yarn_app_logs_path, app_id, container_id, 'stderr')
                with open(log_file_path, 'r') as f:
                    first_line = f.readline()
                    if 'dmlc.ApplicationMaster' not in first_line:
                        logging.error(f.read())
            if not os.path.exists(OUTPUT_FAILED_FILE):
                with open(OUTPUT_FAILED_FILE, "w") as f:
                    f.write("Customer Error: Out of Memory. Please use a larger "
                            "instance and/or reduce the values of other parameters "
                            "(e.g. num_classes.) if applicable")
            sys.exit(1)
        except Exception as e:
            logging.exception(e)


def start_yarn_daemons(num_hosts, current_host, master_host, master_ip):
    file_prepare()
    cluster_config(num_hosts, current_host, master_host, master_ip)
    start_daemons(master_host, current_host)


def get_ip_from_host(host_name):
    IP_WAIT_TIME = 300
    counter = 0
    ip = ''

    while counter < IP_WAIT_TIME and ip == '':
        try:
            ip = socket.gethostbyname(host_name)
            break
        except:  # noqa: E722
            counter += 1
            time.sleep(1)

    if counter == IP_WAIT_TIME and ip == '':
        raise PlatformError("Network issue happened. Cannot retrieve ip address in past 10 minutes")

    return ip


def get_all_sizes():
    TRAIN_CHANNEL = 'train'
    VAL_CHANNEL = 'validation'
    train_path = INPUT_DATA_PATH + '/' + TRAIN_CHANNEL
    val_path = INPUT_DATA_PATH + '/' + VAL_CHANNEL

    train_files_size = convertToMb(get_size(train_path))
    val_files_size = convertToMb(get_size(val_path))
    mem_size = convertToMb(psutil.virtual_memory().available)

    return train_files_size, val_files_size, mem_size


def convertToMb(size):
    val = str(round((size) / (1024 * 1024), 2)) + 'mb'
    return val
