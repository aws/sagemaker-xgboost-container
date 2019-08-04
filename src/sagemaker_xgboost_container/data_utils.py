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

import xgboost as xgb

from sagemaker_algorithm_toolkit import exceptions as exc
from sagemaker_containers import _content_types
from sagemaker_xgboost_container.constants import xgb_content_types


CSV = 'csv'
LIBSVM = 'libsvm'


def get_content_type(content_type_cfg_val):
    """Get content type from data config.

    Assumes that training and validation data have the same content type.

    ['libsvm', 'text/libsvm', 'text/x-libsvm'] will return 'libsvm'
    ['csv', 'text/csv'] will return 'csv'

    :param content_type_cfg_val
    :return: Parsed content type
    """
    if content_type_cfg_val is None:
        return LIBSVM
    elif content_type_cfg_val.lower() in [LIBSVM, xgb_content_types.LIBSVM, xgb_content_types.X_LIBSVM]:
        return LIBSVM
    elif content_type_cfg_val.lower() in [CSV, _content_types.CSV]:
        return CSV
    else:
        raise exc.UserError("ContentType should be one of these options:"
                            " 'csv', 'libsvm', 'text/csv', 'text/libsvm', 'text/x-libsvm'.")


def _is_data_file(file_path, file_name):
    """Return true if file name is a valid data file name.

    A file is valid if:
    * File name does not start with '.' or '_'.
    * File is not a XGBoost cache file.

    :param file_path:
    :param file_name:
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


def get_csv_dmatrix(files_path, csv_weights):
    """Get Data Matrix from CSV files.

    :param files_path: File path where CSV formatted training data resides, either directory or file
    :param csv_weights:
    :return: xgb.DMatrix
    """
    # infer delimiter of CSV input
    csv_file = files_path if os.path.isfile(files_path) else [
        f for f in os.listdir(files_path) if os.path.isfile(os.path.join(files_path, f))][0]
    with open(os.path.join(files_path, csv_file), errors='ignore') as f:
        sample_text = f.readline().strip()
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


def get_libsvm_dmatrix(files_path):
    """Get DMatrix from libsvm file path.

    :param files_path: File path where LIBSVM formatted training data resides, either directory or file
    :return: xgb.DMatrix
    """
    try:
        dmatrix = xgb.DMatrix(files_path)
    except Exception as e:
        raise exc.UserError("Failed to load libsvm data with exception:\n{}".format(e))

    return dmatrix


def get_dmatrix(data_path, file_type, csv_weights=0):
    """Create Data Matrix from CSV or LIBSVM file.

    :param data_path: Either directory or file
    :param file_type:
    :param csv_weights: Only used if file_type is 'csv'.
                        1 if the instance weights are in the second column of csv file; otherwise, 0
    :return: xgb.DMatrix
    """
    if not os.path.exists(data_path):
        return None
    else:
        if os.path.isfile(data_path):
            files_path = data_path
        else:
            for root, dirs, files in os.walk(data_path):
                if dirs == []:
                    files_path = root
                    break
        if file_type.lower() == CSV:
            dmatrix = get_csv_dmatrix(files_path, csv_weights)
        elif file_type.lower() == LIBSVM:
            dmatrix = get_libsvm_dmatrix(files_path)

        if dmatrix.get_label().size == 0:
            raise exc.UserError(
                "Got input data without labels. Please check the input data set. "
                "If training job is running on multiple instances, please switch "
                "to using single instance if number of records in the data set "
                "is less than number of workers (16 * number of instance) in the cluster.")

    return dmatrix


def get_size(data_path):
    """Return size of data files at dir_path.

    :param data_path: Either directory or file
    :return: Size of data
    """
    if not os.path.exists(data_path):
        logging.info('Path {} does not exist!'.format(data_path))
        return 0
    else:
        total_size = 0
        if os.path.isfile(data_path):
            return os.path.getsize(data_path)
        else:
            for root, dirs, files in os.walk(data_path):
                for current_file in files:
                    if current_file.startswith('.'):
                        raise exc.UserError("Hidden file found in the data path! Remove that before training.")
                    file_path = os.path.join(root, current_file)
                    total_size += os.path.getsize(file_path)
            return total_size
