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
import pandas as pd
import re

import numpy as np
from scipy.sparse import vstack
from sklearn.datasets import load_svmlight_files

import xgboost as xgb

from sagemaker_containers import _content_types

from sagemaker_algorithm_toolkit import exceptions as exc
from sagemaker_xgboost_container.constants import xgb_content_types
from sagemaker_xgboost_container.metrics.custom_metrics import get_custom_metrics, configure_feval

HPO_SEPARATOR = ':'

CSV = 'csv'
LIBSVM = 'libsvm'


# These are helper functions for parsing data
def _valid_file(file_path, file_name):
    """Validate data file.

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
    """Validate file format by its contents"""
    if not os.path.exists(dir_path):
        return
    else:
        files_path = None
        for root, dirs, files in os.walk(dir_path):
            if dirs == []:
                files_path = root
                break
        data_files = [f for f in os.listdir(files_path) if _valid_file(files_path, f)]
        if file_type.lower() == CSV:
            for data_file in data_files:
                _validate_csv_format(os.path.join(files_path, data_file))
        elif file_type.lower() == LIBSVM:
            for data_file in data_files:
                _validate_libsvm_format(os.path.join(files_path, data_file))


def _validate_csv_format(file_path):
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


def _validate_libsvm_format(file_path):
    with open(file_path, 'r', errors='ignore') as f:
        first_line = f.readline()
        # validate it's not csv
        if not (' ' in first_line and ':' in first_line):
            raise exc.UserError("Blankspace and colon not found in firstline '{}...' of file '{}'. ContentType by "
                                "defaullt is in libsvm. Please ensure the file is in libsvm format.".format(
                                    first_line[:50], file_path.split('/')[-1]))


def get_dmatrix(dir_path, file_type, is_distributed, is_training, csv_weights=0):
    """Create DataMatrix from csv or libsvm file.

    :param dir_path:
    :param file_type:
    :param is_distributed:
    :param is_training:
    :param csv_weights: If true, parse weights from column 1
    :return: xgb.DMatrix
    """
    if not os.path.exists(dir_path):
        return None
    else:
        files_path = None
        for root, dirs, files in os.walk(dir_path):
            if dirs == []:
                files_path = root
                break
        if file_type.lower() == CSV:
            dmatrix = get_csv_dmatrix(files_path, is_training, csv_weights)
        elif file_type.lower() == LIBSVM:
            dmatrix = get_libsvm_dmatrix(files_path, is_distributed)

        if dmatrix.get_label().size == 0:
            raise exc.UserError(
                "Got input data without labels. Please check the input data set. "
                "If training job is running on multiple instances, please switch "
                "to using single instance if number of records in the data set "
                "is less than number of workers (16 * number of instance) in the cluster.")

    return dmatrix


def _csv_file_pop_first_line(file_path):
    with open(file_path, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(file_path, 'w') as fout:
        pop_line = data[:1]
        fout.writelines(data[1:])
    return pop_line


def _get_formatted_csv_file_path(files_path, delimiter, weights=0):
    base_path_str = '{}?format=csv&label_column=0&delimiter={}'.format(files_path, delimiter)
    if weights == 1:
        return '{}&weight_column=1'.format(base_path_str)
    else:
        return base_path_str


def _get_dmatrix_format_header(df):
    return ['f{}'.format(idx) for idx in range(len(df.columns))]


def get_csv_dmatrix(files_path, is_training, csv_weights):
    """Get Data Matrix from CSV files.

    If not distributed, use naive DMatrix initialize.

    Note: If distributed, read CSV files into pandas DataFrame. Also manually parse first line in CSV
    files to initialize DMatrix with weights. This is to avoid the sharding DMatrix does
    when loading from file in xgboost=0.90.0:

        https://github.com/dmlc/xgboost/blob/a22368d2100d8f964c1c47fe8c04bfbe17b65060/src/c_api/c_api.cc#L239

    TODO: Just use native DMatrix file load when it supports ability to turn off sharding in distributed mode

    :param files_path: Data filepath
    :param is_training: Boolean to indicate training
    :param csv_weights:
    :return: xgb.DMatrix
    """
    # infer delimiter of CSV input
    csv_file = [f for f in os.listdir(files_path) if os.path.isfile(os.path.join(files_path, f))][0]
    with open(os.path.join(files_path, csv_file), errors='ignore') as f:
        sample_text = f.readline().strip()
    try:
        delimiter = csv.Sniffer().sniff(sample_text).delimiter
        logging.info("Determined delimiter of CSV input is \'{}\'".format(delimiter))
    except Exception as e:
        raise exc.UserError("Could not determine delimiter on line {}:\n{}".format(sample_text[:50], e))

    try:
        data_files = sorted([os.path.join(files_path, f) for f in os.listdir(files_path)])
        parsed_csv_weights = None

        train_data = pd.concat([pd.read_csv(data_file, sep=delimiter, header=None) for data_file in data_files])
        train_label = train_data.iloc[:, 0]
        train_df = train_data.iloc[:, 1:]

        if is_training and csv_weights == 1:
            parsed_csv_weights = train_df.iloc[:, 0]
            train_df = train_df.iloc[:, 1:]

        train_df.columns = _get_dmatrix_format_header(train_df)

        dmatrix = xgb.DMatrix(data=train_df, label=train_label, weight=parsed_csv_weights)
        # else:
        #     if csv_weights == 1:
        #         dmatrix = xgb.DMatrix(
        #             '{}?format=csv&label_column=0&delimiter={}&weight_column=1'.format(files_path, delimiter))
        #     else:
        #         dmatrix = xgb.DMatrix('{}?format=csv&label_column=0&delimiter={}'.format(files_path, delimiter))

    except Exception as e:
        raise exc.UserError("Failed to load csv data with exception:\n{}".format(e))
    return dmatrix


def _libsvm_label_has_weights(libsvm_files):
    libsvm_file = libsvm_files[0]
    with open(libsvm_file, errors='ignore') as f:
        first_libsvm_line = f.readline()
        label = first_libsvm_line.split(' ', 2)[0]
        if ':' in label:
            return True
        else:
            return False


def _parse_weights_from_libsvm(libsvm_files):
    weights = []

    for libsvm_file in libsvm_files:
        with open(libsvm_file, errors='ignore') as f_read:
            libsvm_content = f_read.readlines()

        with open(libsvm_file, mode='w', errors='ignore') as f_write:
            for libsvm_line in libsvm_content:
                parsed_line = libsvm_line.split(' ', maxsplit=1)
                label = parsed_line[0]
                features = parsed_line[1]

                split_label = label.split(':', maxsplit=1)
                parsed_label = split_label[0]
                parsed_weight = split_label[1]

                weights.append(float(parsed_weight))
                new_libsvm_line = "{} {}".format(parsed_label, features)

                f_write.write(new_libsvm_line)

    return weights


def get_libsvm_dmatrix(files_path, is_distributed):
    """Get DMatrix from libsvm file path.

    NOTE: In distributed mode, the data is first parsed into CSR matrix. This is to avoid the sharding DMatrix does
    when loading from file in xgboost=0.90.0:

        https://github.com/dmlc/xgboost/blob/a22368d2100d8f964c1c47fe8c04bfbe17b65060/src/c_api/c_api.cc#L239

    TODO: Just use native DMatrix file load when it supports ability to turn off sharding in distributed mode

    :param files_path: File path where training data resides
    :param is_distributed: True if distributed training.
            If true, this will cause libsvm file to be read into scipy.sparse.csr_matrix before initializing DMatrix.
            If false, DMatrix will get the {rabit.get_rank()}th of (rabit.get_world_size()) partitions
                of the data in files_path
    :return: xgb.DMatrix
    """
    try:
        if is_distributed:
            files_to_load = [os.path.join(files_path, file_name) for file_name in os.listdir(files_path)]

            has_weights = _libsvm_label_has_weights(files_to_load)
            if has_weights:
                weights = _parse_weights_from_libsvm(files_to_load)
            else:
                weights = None

            csr_matrix_list = load_svmlight_files(files_to_load, zero_based=True)

            labels = []
            data_matrices = []

            # sklearn.datasets.load_svmlight_files() will return [X1, y1, X2, y2...], so iterate items two at a time
            sparse_matrix_iter = iter(csr_matrix_list)
            for next_sparse_matrix in sparse_matrix_iter:
                next_labels = next(sparse_matrix_iter)
                data_matrices.append(next_sparse_matrix)
                labels.append(next_labels)

            combined_matrix = vstack(data_matrices)
            flattened_labels = [item for sublist in labels for item in sublist]

            dmatrix = xgb.DMatrix(data=combined_matrix, label=flattened_labels, weight=weights)
        else:
            dmatrix = xgb.DMatrix(files_path)
    except Exception as e:
        raise exc.UserError("Failed to load libsvm data with exception:\n{}".format(e))

    return dmatrix


def get_size_validated_dmatrix(train_path, validate_path, file_type, is_distributed, csv_weights=0):
    train_files_size = get_size(train_path)
    if validate_path:
        val_files_size = get_size(validate_path)
    else:
        val_files_size = 0

    logging.debug("File size need to be processed in the node: {}mb.".format(
        round((train_files_size + val_files_size) / (1024 * 1024), 2)))

    validate_file_format(train_path, file_type)
    dtrain = get_dmatrix(train_path, file_type, is_distributed=is_distributed,
                         is_training=True, csv_weights=csv_weights)

    if validate_path:
        validate_file_format(validate_path, file_type)
        dval = get_dmatrix(validate_path, file_type, is_distributed=is_distributed,
                           is_training=False, csv_weights=csv_weights)
    else:
        dval = None

    return dtrain, dval


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
    content_type = data_cfg['train'].get("ContentType")

    if content_type is None:
        logging.info("Content-type is not set, defaulting to: txt/libsvm")
    if content_type.lower() in [LIBSVM, xgb_content_types.LIBSVM, xgb_content_types.X_LIBSVM]:
        return LIBSVM
    elif content_type.lower() in [CSV, _content_types.CSV]:
        return CSV
    else:
        raise exc.UserError("ContentType should be one of these options: '{}', '{}', '{}'.".format(
             _content_types.CSV, LIBSVM, xgb_content_types.X_LIBSVM))


# These are helper functions for parsing the list of metrics to be outputted
def get_union_metrics(metric_a, metric_b):
    """Union metric_a and metric_b

    :param metric_a: list
    :param metric_b: list
    :return: union metrics list from metric_a and metric_b
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


def _get_bytes_to_mb(num_bytes):
    return round(num_bytes / (1024 * 1024), 2)
