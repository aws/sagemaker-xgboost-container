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


from sagemaker_algorithm_toolkit import exceptions as exc
from sagemaker_xgboost_container.bootstrap import file_prepare, cluster_config, start_daemons
from sagemaker_xgboost_container.constants.xgb_constants import LOGISTIC_REGRESSION_LABEL_RANGE_ERROR, \
    MULTI_CLASS_LABEL_RANGE_ERROR, FEATURE_MISMATCH_ERROR, LABEL_PREDICTION_SIZE_MISMATCH, ONLY_POS_OR_NEG_SAMPLES, \
    BASE_SCORE_RANGE_ERROR, POISSON_REGRESSION_ERROR, FNULL, TWEEDIE_REGRESSION_ERROR, REG_LAMBDA_ERROR
from sagemaker_xgboost_container.metrics.custom_metrics import CUSTOM_METRICS, get_custom_metrics, configure_feval


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
            raise exc.UserError(
                "Blankspace and colon found in firstline '{}...' of file '{}'. Please ensure the file is in csv "
                "format.".format(
                    first_line[:50], file_path.split('/')[-1]))
        # validate no header line
        match_object = re.search('[a-df-zA-DF-Z]', first_line)
        if match_object is not None:
            raise exc.UserError("Non-numeric value '{}' found in the header line '{}...' of file '{}'. "
                                "CSV format require no header line in it. If header line is already removed, "
                                "XGBoost does not accept non-numeric value in the data.".format(
                                    match_object.group(0), first_line[:50], file_path.split('/')[-1]))


def validate_libsvm_format(file_path):
    with open(file_path, 'r', errors='ignore') as f:
        first_line = f.readline()
        # validate it's not csv
        if not (' ' in first_line and ':' in first_line):
            raise exc.UserError("Blankspace and colon not found in firstline '{}...' of file '{}'. ContentType by "
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
        raise exc.UserError("Could not determine delimiter on line {}:\n{}".format(sample_text[:50], e))

    try:
        if csv_weights == 1:
            dmatrix = xgb.DMatrix(
                '{}?format=csv&label_column=0&delimiter={}&weight_column=1'.format(files_path, delimiter))
        else:
            dmatrix = xgb.DMatrix('{}?format=csv&label_column=0&delimiter={}'.format(files_path, delimiter))
    except Exception as e:
        raise exc.UserError("Failed to load csv data with exception:\n{}".format(e))
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
        raise exc.UserError("Failed to load libsvm data with exception:\n{}".format(e))
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
                raise exc.UserError("Insufficient memory available in the instance. External memory for csv input not "
                                    "supported.\
                                 Switch to larger instance with more RAM/distributed mode for training.")
            dmatrix = get_csv_dmatrix(files_path, csv_weights)
        elif file_type.lower() == 'libsvm':
            dmatrix = get_libsvm_dmatrix(files_path, exceed_memory)

        if dmatrix.get_label().size == 0:
            raise exc.UserError(
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
                    raise exc.UserError("Hidden file found in the data path! Remove that before training.")
                fp = os.path.join(root, f)
                total_size += os.path.getsize(fp)
        if total_size == 0:
            raise exc.UserError(
                "No file found in path {}. If you are using ShardedByS3Key, make sure every machine get at least one "
                "partition file".format(
                    dir_path))
        return total_size


def get_content_type(data_cfg):
    tmp = data_cfg[TRAIN_CHANNEL].get("ContentType")
    file_type = None
    if tmp is None or tmp.lower() == 'libsvm' or tmp.lower() == 'text/libsvm':
        file_type = 'libsvm'
    elif tmp.lower() == 'csv' or tmp.lower() == 'text/csv':
        file_type = 'csv'
    else:
        raise exc.UserError("ContentType should be one of these options: 'text/libsvm', 'text/csv'.")
    return file_type


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


def get_eval_metrics_and_feval(tuning_objective_metric_param, eval_metric):
    """Return list of default xgb evaluation metrics and list of container defined metrics.

    XGB uses the 'eval_metric' parameter for the evaluation metrics supported by default, and 'feval' as an argument
    during training to validate using custom evaluation metrics. The argument 'feval' takes a function as value; the
    method returned here will be configured to run for only the metrics the user specifies.

    :param tuning_objective_metric_param: HPO metric
    :param eval_metric: list of xgb metrics to output
    :return: cleaned list of xgb supported evaluation metrics, method configured with container defined metrics.
    """
    tuning_objective_metric = None
    configured_eval = None
    cleaned_eval_metrics = None

    if tuning_objective_metric_param is not None:
        tuning_objective_metric_tuple = MetricNameComponents.decode(tuning_objective_metric_param)
        tuning_objective_metric = tuning_objective_metric_tuple.metric_name.split(',')
        logging.info('Setting up HPO optimized metric to be : {}'.format(tuning_objective_metric_tuple.metric_name))

    union_metrics = get_union_metrics(tuning_objective_metric, eval_metric)

    if union_metrics is not None:
        feval_metrics = get_custom_metrics(union_metrics)
        if feval_metrics:
            configured_eval = configure_feval(feval_metrics)
            cleaned_eval_metrics = list(set(union_metrics) - set(feval_metrics))
        else:
            cleaned_eval_metrics = union_metrics

    return cleaned_eval_metrics, configured_eval


class MetricNameComponents(object):
    def __init__(self, data_segment, metric_name, emission_frequency=None):
        self.data_segment = data_segment
        self.metric_name = metric_name
        self.emission_frequency = emission_frequency

    @classmethod
    def decode(cls, tuning_objective_metric):
        result = tuning_objective_metric.split(":")
        return MetricNameComponents(*result)


def train_job(resource_config, train_cfg, data_cfg):
    param = train_cfg
    if param.get("updater"):
        param["updater"] = ",".join(param["updater"])

    early_stopping_rounds = train_cfg["early_stopping_rounds"]
    num_round = train_cfg["num_round"]
    csv_weights = train_cfg["csv_weights"]

    tuning_objective_metric_param = param.get("_tuning_objective_metric")
    eval_metric = param.get("eval_metric")
    cleaned_eval_metric, configured_feval = get_eval_metrics_and_feval(tuning_objective_metric_param, eval_metric)
    if cleaned_eval_metric:
        param['eval_metric'] = cleaned_eval_metric
    else:
        param.pop('eval_metric', None)

    num_hosts = len(resource_config["hosts"])
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
        bst = xgb.train(param, dtrain, num_boost_round=num_round, evals=watchlist, feval=configured_feval,
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
                    f.write("User Error: Out of Memory. Please use a larger "
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
        raise exc.PlatformError("Network issue happened. Cannot retrieve ip address in past 10 minutes")

    return ip
