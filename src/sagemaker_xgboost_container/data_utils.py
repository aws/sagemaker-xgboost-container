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
import re

import xgboost as xgb

from sagemaker_algorithm_toolkit import exceptions as exc


CSV = 'csv'
LIBSVM = 'libsvm'


def get_content_type(data_cfg):
    tmp = data_cfg['train'].get("ContentType")
    file_type = None
    if tmp is None or tmp.lower() == 'libsvm' or tmp.lower() == 'text/libsvm':
        file_type = 'libsvm'
    elif tmp.lower() == 'csv' or tmp.lower() == 'text/csv':
        file_type = 'csv'
    else:
        raise exc.UserError("ContentType should be one of these options: 'text/libsvm', 'text/csv'.")
    return file_type


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


def get_dmatrix(dir_path, file_type, csv_weights=0):
    """Create DataMatrix from csv or libsvm file.

    :param dir_path:
    :param file_type:
    :param csv_weights:
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


def get_csv_dmatrix(files_path, csv_weights):
    """Get Data Matrix from CSV files.

    :param files_path: Data filepath
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

    :param files_path: File path where training data resides
    :return: xgb.DMatrix
    """
    try:
        dmatrix = xgb.DMatrix(files_path)
    except Exception as e:
        raise exc.UserError("Failed to load libsvm data with exception:\n{}".format(e))

    return dmatrix


def get_size_validated_dmatrix(train_path, validate_path, file_type, csv_weights=0):
    if train_path:
        train_files_size = get_size(train_path)
    else:
        train_files_size = 0

    if validate_path:
        val_files_size = get_size(validate_path)
    else:
        val_files_size = 0

    logging.debug("File size need to be processed in the node: {}mb.".format(
        round((train_files_size + val_files_size) / (1024 * 1024), 2)))

    if train_files_size > 0:
        validate_file_format(train_path, file_type)
        dtrain = get_dmatrix(train_path, file_type, csv_weights=csv_weights)
    else:
        dtrain = None

    if val_files_size > 0:
        validate_file_format(validate_path, file_type)
        dval = get_dmatrix(validate_path, file_type)
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
        return total_size
