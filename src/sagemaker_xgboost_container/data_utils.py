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
import cgi
import csv
import logging
import os

import mlio
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

from sagemaker_algorithm_toolkit import exceptions as exc
from sagemaker_containers import _content_types
from sagemaker_xgboost_container.constants import xgb_content_types

from mlio.integ.numpy import as_numpy
from mlio.integ.arrow import as_arrow_file

BATCH_SIZE = 4000

CSV = 'csv'
LIBSVM = 'libsvm'
PARQUET = 'parquet'
RECORDIO_PROTOBUF = 'recordio-protobuf'

VALID_CONTENT_TYPES = [CSV, LIBSVM, PARQUET, RECORDIO_PROTOBUF,
                       _content_types.CSV, xgb_content_types.LIBSVM,
                       xgb_content_types.X_LIBSVM, xgb_content_types.X_PARQUET,
                       xgb_content_types.X_RECORDIO_PROTOBUF]

VALID_PIPED_CONTENT_TYPES = [CSV, PARQUET, RECORDIO_PROTOBUF,
                             _content_types.CSV, xgb_content_types.X_PARQUET,
                             xgb_content_types.X_RECORDIO_PROTOBUF]


INVALID_CONTENT_TYPE_ERROR = "{invalid_content_type} is not an accepted ContentType: " + \
                             ", ".join(['%s' % c for c in VALID_CONTENT_TYPES]) + "."
INVALID_CONTENT_FORMAT_ERROR = "First line '{line_snippet}...' of file '{file_name}' is not " \
                               "'{content_type}' format. Please ensure the file is in '{content_type}' format."


def _get_invalid_content_type_error_msg(invalid_content_type):
    return INVALID_CONTENT_TYPE_ERROR.format(invalid_content_type=invalid_content_type)


def _get_invalid_libsvm_error_msg(line_snippet, file_name):
    return INVALID_CONTENT_FORMAT_ERROR.format(line_snippet=line_snippet, file_name=file_name, content_type='LIBSVM')


def _get_invalid_csv_error_msg(line_snippet, file_name):
    return INVALID_CONTENT_FORMAT_ERROR.format(line_snippet=line_snippet, file_name=file_name, content_type='CSV')


def _get_csv_content_type(content_type_cfg_val):
    """
    Return CSV if content_type_cfg_val is
    * 'csv',
    * 'text/csv',
    * 'text/csv; ...' with valid parameters
    """
    if content_type_cfg_val in [CSV, _content_types.CSV]:
        # Allow 'csv' and 'text/csv'
        return CSV
    else:
        content_type, params = cgi.parse_header(content_type_cfg_val)
        if content_type == _content_types.CSV:
            if params.get('label_size') is not None and params['label_size'] != '1':
                msg = "{} is not an accepted csv ContentType. "\
                      "Optional parameter label_size must be equal to 1".format(content_type_cfg_val)
                raise exc.UserError(msg)
            return CSV
    raise exc.UserError(_get_invalid_content_type_error_msg(content_type_cfg_val))


def get_content_type(content_type_cfg_val):
    """Get content type from data config.

    Assumes that training and validation data have the same content type.

    ['libsvm', 'text/libsvm', 'text/x-libsvm'] will return 'libsvm'
    ['csv', 'text/csv', 'text/csv; label_size=1'] will return 'csv'

    :param content_type_cfg_val
    :return: Parsed content type
    """
    if content_type_cfg_val is None:
        return LIBSVM
    elif content_type_cfg_val.lower() in [LIBSVM, xgb_content_types.LIBSVM, xgb_content_types.X_LIBSVM]:
        return LIBSVM
    elif CSV in content_type_cfg_val.lower():
        return _get_csv_content_type(content_type_cfg_val.lower())
    elif content_type_cfg_val.lower() in [PARQUET, xgb_content_types.X_PARQUET]:
        return PARQUET
    elif content_type_cfg_val.lower() in [RECORDIO_PROTOBUF, xgb_content_types.X_RECORDIO_PROTOBUF]:
        return RECORDIO_PROTOBUF
    else:
        raise exc.UserError(_get_invalid_content_type_error_msg(content_type_cfg_val))


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


def _get_csv_delimiter(sample_csv_line):
    try:
        delimiter = csv.Sniffer().sniff(sample_csv_line).delimiter
        logging.info("Determined delimiter of CSV input is \'{}\'".format(delimiter))
    except Exception as e:
        raise exc.UserError("Could not determine delimiter on line {}:\n{}".format(sample_csv_line[:50], e))
    return delimiter


def _get_num_valid_libsvm_features(libsvm_line):
    """Get number of valid LIBSVM features.

    XGBoost expects the following LIBSVM format:
        <label>(:<instance weight>) <index>:<value> <index>:<value> <index>:<value> ...

    :param libsvm_line:
    :return: -1 if the line is not a valid LIBSVM line; otherwise, return number of correctly formatted features
    """
    split_line = libsvm_line.split(' ')
    num_sparse_features = 0

    if not _is_valid_libsvm_label(split_line[0]):
        logging.error("{} does not follow LIBSVM label format <label>(:<weight>).".format(split_line[0]))
        return -1

    if len(split_line) > 1:
        for idx in range(1, len(split_line)):
            if ':' not in split_line[idx]:
                return -1
            else:
                libsvm_feature_contents = split_line[1].split(':')
                if len(libsvm_feature_contents) != 2:
                    return -1
                else:
                    num_sparse_features += 1
        return num_sparse_features
    else:
        return 0


def _is_valid_libsvm_label(libsvm_label):
    """Check if LIBSVM label is formatted like so:

    <label> if just label
    <label>:<instance_weight> if label and instance weight both exist

    :param libsvm_label:
    """
    split_label = libsvm_label.split(':')

    if len(split_label) <= 2:
        for label_part in split_label:
            try:
                float(label_part)
            except ValueError:
                return False
    else:
        return False

    return True


def _validate_csv_format(file_path):
    """Validate that data file is CSV format.

    Check that delimiter can be inferred.

    Note: This only validates the first line in the file. This is not a comprehensive file check,
    as XGBoost will have its own data validation.

    :param file_path
    """
    with open(file_path, 'r', errors='ignore') as read_file:
        line_to_validate = read_file.readline()
        _get_csv_delimiter(line_to_validate)


def _validate_libsvm_format(file_path):
    """Validate that data file is LIBSVM format.

    XGBoost expects the following LIBSVM format:
        <label>(:<instance weight>) <index>:<value> <index>:<value> <index>:<value> ...

    Note: This only validates the first line that has a feature. This is not a comprehensive file check,
    as XGBoost will have its own data validation.

    :param file_path
    """
    with open(file_path, 'r', errors='ignore') as read_file:
        for line_to_validate in read_file:
            num_sparse_libsvm_features = _get_num_valid_libsvm_features(line_to_validate)

            if num_sparse_libsvm_features > 1:
                # Return after first valid LIBSVM line with features
                return
            elif num_sparse_libsvm_features < 0:
                raise exc.UserError(_get_invalid_libsvm_error_msg(
                    line_snippet=line_to_validate[:50], file_name=file_path.split('/')[-1]))

    logging.warning("File {} is not an invalid LIBSVM file but has no features. Accepting simple validation.".format(
        file_path.split('/')[-1]))


def validate_data_file_path(data_path, content_type):
    """Validate data in data_path are formatted correctly based on content_type.

    Note: This is not a comprehensive validation. XGBoost has its own content validation.

    :param data_path:
    :param content_type:
    """
    parsed_content_type = get_content_type(content_type)

    if not os.path.exists(data_path):
        raise exc.UserError("{} is not a valid path!".format(data_path))
    else:

        if os.path.isfile(data_path):
            data_files = [data_path]
        else:
            dir_path = None
            for root, dirs, files in os.walk(data_path):
                if dirs == []:
                    dir_path = root
                    break
            data_files = [
                os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path) if _is_data_file(
                    dir_path, file_name)]
        if parsed_content_type.lower() == CSV:
            for data_file_path in data_files:
                _validate_csv_format(data_file_path)
        elif parsed_content_type.lower() == LIBSVM:
            for data_file_path in data_files:
                _validate_libsvm_format(data_file_path)
        elif parsed_content_type.lower() == PARQUET or parsed_content_type.lower() == RECORDIO_PROTOBUF:
            # No op
            return


def _get_csv_dmatrix_file_mode(files_path, csv_weights):
    """Get Data Matrix from CSV data in file mode.

    Infer the delimiter of data from first line of first data file.

    :param files_path: File path where CSV formatted training data resides, either directory or file
    :param csv_weights: 1 if instance weights are in second column of CSV data; else 0
    :return: xgb.DMatrix
    """
    csv_file = files_path if os.path.isfile(files_path) else [
        f for f in os.listdir(files_path) if os.path.isfile(os.path.join(files_path, f))][0]
    with open(os.path.join(files_path, csv_file)) as read_file:
        sample_csv_line = read_file.readline()
    delimiter = _get_csv_delimiter(sample_csv_line)

    try:
        if csv_weights == 1:
            dmatrix = xgb.DMatrix(
                '{}?format=csv&label_column=0&delimiter={}&weight_column=1'.format(files_path, delimiter))
        else:
            dmatrix = xgb.DMatrix('{}?format=csv&label_column=0&delimiter={}'.format(files_path, delimiter))

    except Exception as e:
        raise exc.UserError("Failed to load csv data with exception:\n{}".format(e))

    return dmatrix


def _get_csv_dmatrix_pipe_mode(pipe_path, csv_weights):
    """Get Data Matrix from CSV data in pipe mode.

    :param pipe_path: SageMaker pipe path where CSV formatted training data is piped
    :param csv_weights: 1 if instance weights are in second column of CSV data; else 0
    :return: xgb.DMatrix or None
    """
    try:
        dataset = [mlio.SageMakerPipe(pipe_path)]
        reader = mlio.CsvReader(dataset=dataset,
                                batch_size=BATCH_SIZE,
                                header_row_index=None)
        examples = []
        for example in reader:
            tmp = [as_numpy(feature).squeeze() for feature in example]
            tmp = np.array(tmp)
            if len(tmp.shape) > 1:
                tmp = tmp.T
            else:
                tmp = np.reshape(tmp, (1, tmp.shape[0]))
            examples.append(tmp)

        if examples:
            data = np.vstack(examples)
            del examples

            if csv_weights == 1:
                dmatrix = xgb.DMatrix(data[:, 2:], label=data[:, 0], weights=data[:, 1])
            else:
                dmatrix = xgb.DMatrix(data[:, 1:], label=data[:, 0])

            return dmatrix
        else:
            return None

    except Exception as e:
        raise exc.UserError("Failed to load csv data with exception:\n{}".format(e))


def get_csv_dmatrix(path, csv_weights, is_pipe=False):
    """Get Data Matrix from CSV data.

    :param path: Path where CSV formatted training data resides, either directory, file, or SageMaker pipe
    :param csv_weights: 1 if instance weights are in second column of CSV data; else 0
    :param is_pipe: Boolean to indicate if data is being read in pipe mode
    :return: xgb.DMatrix or None
    """
    if is_pipe:
        return _get_csv_dmatrix_pipe_mode(path, csv_weights)
    else:
        return _get_csv_dmatrix_file_mode(path, csv_weights)


def get_libsvm_dmatrix(files_path, is_pipe=False):
    """Get DMatrix from libsvm file path.

    Pipe mode not currently supported for libsvm.

    :param files_path: File path where LIBSVM formatted training data resides, either directory or file
    :param is_pipe: Boolean to indicate if data is being read in pipe mode
    :return: xgb.DMatrix
    """
    if is_pipe:
        raise exc.UserError("Pipe mode not supported for LibSVM.")

    try:
        dmatrix = xgb.DMatrix(files_path)
    except Exception as e:
        raise exc.UserError("Failed to load libsvm data with exception:\n{}".format(e))

    return dmatrix


def _get_parquet_dmatrix_file_mode(files_path):
    """Get Data Matrix from parquet data in file mode.

    :param files_path: File path where parquet formatted training data resides, either directory or file
    :return: xgb.DMatrix
    """
    try:
        table = pq.read_table(files_path)

        data = table.to_pandas()
        del table

        if type(data) is pd.DataFrame:
            # pyarrow.Table.to_pandas may produce NumPy array or pandas DataFrame
            data = data.to_numpy()

        dmatrix = xgb.DMatrix(data[:, 1:], label=data[:, 0])
        del data

        return dmatrix

    except Exception as e:
        raise exc.UserError("Failed to load parquet data with exception:\n{}".format(e))


def _get_parquet_dmatrix_pipe_mode(pipe_path):
    """Get Data Matrix from parquet data in pipe mode.

    :param pipe_path: SageMaker pipe path where parquet formatted training data is piped
    :return: xgb.DMatrix or None
    """
    try:
        f = mlio.SageMakerPipe(pipe_path)
        examples = []

        with f.open_read() as strm:
            reader = mlio.ParquetRecordReader(strm)

            for record in reader:
                table = pq.read_table(as_arrow_file(record))
                array = table.to_pandas()
                if type(array) is pd.DataFrame:
                    array = array.to_numpy()
                examples.append(array)

        if examples:
            data = np.vstack(examples)
            del examples

            dmatrix = xgb.DMatrix(data[:, 1:], label=data[:, 0])
            return dmatrix
        else:
            return None

    except Exception as e:
        raise exc.UserError("Failed to load parquet data with exception:\n{}".format(e))


def get_parquet_dmatrix(path, is_pipe=False):
    """Get Data Matrix from parquet data.

    :param path: Path where parquet formatted training data resides, either directory, file, or SageMaker pipe
    :param is_pipe: Boolean to indicate if data is being read in pipe mode
    :return: xgb.DMatrix or None
    """
    if is_pipe:
        return _get_parquet_dmatrix_pipe_mode(path)
    else:
        return _get_parquet_dmatrix_file_mode(path)


def get_recordio_protobuf_dmatrix(path, is_pipe=False):
    """Get Data Matrix from recordio-protobuf data.

    :param path: Path where recordio-protobuf formatted training data resides, either directory, file, or SageMaker pipe
    :param is_pipe: Boolean to indicate if data is being read in pipe mode
    :return: xgb.DMatrix or None
    """
    try:
        if is_pipe:
            dataset = [mlio.SageMakerPipe(path)]
            reader = mlio.RecordIOProtobufReader(dataset=dataset,
                                                 batch_size=BATCH_SIZE)
        else:
            dataset = mlio.list_files(path)
            reader = mlio.RecordIOProtobufReader(dataset=dataset,
                                                 batch_size=BATCH_SIZE)

        examples = []
        for example in reader:
            tmp = [as_numpy(feature) for feature in example]
            tmp = np.hstack(tmp)
            examples.append(tmp)

        if examples:
            data = np.vstack(examples)
            del examples

            dmatrix = xgb.DMatrix(data[:, 1:], label=data[:, 0])
            return dmatrix
        else:
            return None

    except Exception as e:
        raise exc.UserError("Failed to load recordio-protobuf data with exception:\n{}".format(e))


def get_dmatrix(data_path, content_type, csv_weights=0, is_pipe=False):
    """Create Data Matrix from CSV or LIBSVM file.

    Assumes that sanity validation for content type has been done.

    :param data_path: Either directory or file
    :param content_type:
    :param csv_weights: Only used if file_type is 'csv'.
                        1 if the instance weights are in the second column of csv file; otherwise, 0
    :param is_pipe: Boolean to indicate if data is being read in pipe mode
    :return: xgb.DMatrix or None
    """
    if not (os.path.exists(data_path) or (is_pipe and os.path.exists(data_path + '_0'))):
        return None
    else:
        if os.path.isfile(data_path) or is_pipe:
            files_path = data_path
        elif not is_pipe:
            for root, dirs, files in os.walk(data_path):
                if dirs == []:
                    files_path = root
                    break
        if content_type.lower() == CSV:
            dmatrix = get_csv_dmatrix(files_path, csv_weights, is_pipe)
        elif content_type.lower() == LIBSVM:
            dmatrix = get_libsvm_dmatrix(files_path, is_pipe)
        elif content_type.lower() == PARQUET:
            dmatrix = get_parquet_dmatrix(files_path, is_pipe)
        elif content_type.lower() == RECORDIO_PROTOBUF:
            dmatrix = get_recordio_protobuf_dmatrix(files_path, is_pipe)

        if dmatrix and dmatrix.get_label().size == 0:
            raise exc.UserError(
                "Got input data without labels. Please check the input data set. "
                "If training job is running on multiple instances, please switch "
                "to using single instance if number of records in the data set "
                "is less than number of workers (16 * number of instance) in the cluster.")

    return dmatrix


def get_size(data_path, is_pipe=False):
    """Return size of data files at dir_path.

    :param data_path: Either directory or file
    :param is_pipe: Boolean to indicate if data is being read in pipe mode
    :return: Size of data or 1 if sagemaker pipe found
    """
    if is_pipe and os.path.exists(data_path + '_0'):
        logging.info('Pipe path {} found.'.format(data_path))
        return 1
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
