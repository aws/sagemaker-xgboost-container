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
import shutil
from typing import List, Union

import mlio
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from mlio.integ.arrow import as_arrow_file
from mlio.integ.numpy import as_numpy
from mlio.integ.scipy import to_coo_matrix
from sagemaker_containers import _content_types
from scipy.sparse import vstack as scipy_vstack

from sagemaker_algorithm_toolkit import exceptions as exc
from sagemaker_xgboost_container.constants import xgb_content_types

BATCH_SIZE = 4000

CSV = "csv"
LIBSVM = "libsvm"
PARQUET = "parquet"
RECORDIO_PROTOBUF = "recordio-protobuf"

MAX_FOLDER_DEPTH = 3

VALID_CONTENT_TYPES = [
    CSV,
    LIBSVM,
    PARQUET,
    RECORDIO_PROTOBUF,
    _content_types.CSV,
    xgb_content_types.LIBSVM,
    xgb_content_types.X_LIBSVM,
    xgb_content_types.X_PARQUET,
    xgb_content_types.X_RECORDIO_PROTOBUF,
]

VALID_PIPED_CONTENT_TYPES = [
    CSV,
    PARQUET,
    RECORDIO_PROTOBUF,
    _content_types.CSV,
    xgb_content_types.X_PARQUET,
    xgb_content_types.X_RECORDIO_PROTOBUF,
]


INVALID_CONTENT_TYPE_ERROR = (
    "{invalid_content_type} is not an accepted ContentType: " + ", ".join(["%s" % c for c in VALID_CONTENT_TYPES]) + "."
)
INVALID_CONTENT_FORMAT_ERROR = (
    "First line '{line_snippet}...' of file '{file_name}' is not "
    "'{content_type}' format. Please ensure the file is in '{content_type}' format."
)


def _get_invalid_content_type_error_msg(invalid_content_type):
    return INVALID_CONTENT_TYPE_ERROR.format(invalid_content_type=invalid_content_type)


def _get_invalid_libsvm_error_msg(line_snippet, file_name):
    return INVALID_CONTENT_FORMAT_ERROR.format(line_snippet=line_snippet, file_name=file_name, content_type="LIBSVM")


def _get_invalid_csv_error_msg(line_snippet, file_name):
    return INVALID_CONTENT_FORMAT_ERROR.format(line_snippet=line_snippet, file_name=file_name, content_type="CSV")


def get_content_type(content_type_cfg_val):
    """Get content type from data config.

    Assumes that training and validation data have the same content type.

    ['libsvm', 'text/libsvm ;charset=utf8', 'text/x-libsvm'] will return 'libsvm'
    ['csv', 'text/csv', 'text/csv; label_size=1'] will return 'csv'

    :param content_type_cfg_val
    :return: Parsed content type
    """
    if content_type_cfg_val is None:
        return LIBSVM
    else:
        # cgi.parse_header extracts all arguments after ';' as key-value pairs
        # e.g. cgi.parse_header('text/csv;label_size=1;charset=utf8') returns
        # the tuple ('text/csv', {'label_size': '1', 'charset': 'utf8'})
        content_type, params = cgi.parse_header(content_type_cfg_val.lower())

        if content_type in [CSV, _content_types.CSV]:
            # CSV content type allows a label_size parameter
            # that should be 1 for XGBoost
            if params and "label_size" in params and params["label_size"] != "1":
                msg = (
                    "{} is not an accepted csv ContentType. "
                    "Optional parameter label_size must be equal to 1".format(content_type_cfg_val)
                )
                raise exc.UserError(msg)
            return CSV
        elif content_type in [LIBSVM, xgb_content_types.LIBSVM, xgb_content_types.X_LIBSVM]:
            return LIBSVM
        elif content_type in [PARQUET, xgb_content_types.X_PARQUET]:
            return PARQUET
        elif content_type in [RECORDIO_PROTOBUF, xgb_content_types.X_RECORDIO_PROTOBUF]:
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
    if file_name.startswith(".") or file_name.startswith("_"):
        return False
    # avoid XGB cache file
    if ".cache" in file_name:
        if "dtrain" in file_name or "dval" in file_name:
            return False
    return True


def _get_csv_delimiter(sample_csv_line):
    try:
        delimiter = csv.Sniffer().sniff(sample_csv_line).delimiter
        logging.info("Determined delimiter of CSV input is '{}'".format(delimiter))
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
    split_line = libsvm_line.split(" ")
    num_sparse_features = 0

    if not _is_valid_libsvm_label(split_line[0]):
        logging.error("{} does not follow LIBSVM label format <label>(:<weight>).".format(split_line[0]))
        return -1

    if len(split_line) > 1:
        for idx in range(1, len(split_line)):
            if ":" not in split_line[idx]:
                return -1
            else:
                libsvm_feature_contents = split_line[1].split(":")
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
    split_label = libsvm_label.split(":")

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
    with open(file_path, "r", errors="ignore") as read_file:
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
    with open(file_path, "r", errors="ignore") as read_file:
        for line_to_validate in read_file:
            num_sparse_libsvm_features = _get_num_valid_libsvm_features(line_to_validate)

            if num_sparse_libsvm_features > 1:
                # Return after first valid LIBSVM line with features
                return
            elif num_sparse_libsvm_features < 0:
                raise exc.UserError(
                    _get_invalid_libsvm_error_msg(
                        line_snippet=line_to_validate[:50], file_name=file_path.split("/")[-1]
                    )
                )

    logging.warning(
        "File {} is not an invalid LIBSVM file but has no features. Accepting simple validation.".format(
            file_path.split("/")[-1]
        )
    )


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
                os.path.join(dir_path, file_name)
                for file_name in os.listdir(dir_path)
                if _is_data_file(dir_path, file_name)
            ]
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
    csv_file = (
        files_path
        if os.path.isfile(files_path)
        else [f for f in os.listdir(files_path) if os.path.isfile(os.path.join(files_path, f))][0]
    )
    with open(os.path.join(files_path, csv_file)) as read_file:
        sample_csv_line = read_file.readline()
    delimiter = _get_csv_delimiter(sample_csv_line)

    try:
        if csv_weights == 1:
            dmatrix = xgb.DMatrix(
                "{}?format=csv&label_column=0&delimiter={}&weight_column=1".format(files_path, delimiter)
            )
        else:
            dmatrix = xgb.DMatrix("{}?format=csv&label_column=0&delimiter={}".format(files_path, delimiter))

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
        pipes_path = pipe_path if isinstance(pipe_path, list) else [pipe_path]
        dataset = [mlio.SageMakerPipe(path) for path in pipes_path]
        reader_params = mlio.DataReaderParams(dataset=dataset, batch_size=BATCH_SIZE)
        csv_params = mlio.CsvParams(header_row_index=None)
        reader = mlio.CsvReader(reader_params, csv_params)

        # Check if data is present in reader
        if reader.peek_example() is not None:
            examples = []
            for example in reader:
                # Write each feature (column) of example into a single numpy array
                tmp = [as_numpy(feature).squeeze() for feature in example]
                tmp = np.array(tmp)
                if len(tmp.shape) > 1:
                    # Columns are written as rows, needs to be transposed
                    tmp = tmp.T
                else:
                    # If tmp is a 1-D array, it needs to be reshaped as a matrix
                    tmp = np.reshape(tmp, (1, tmp.shape[0]))
                examples.append(tmp)

            data = np.vstack(examples)
            del examples

            if csv_weights == 1:
                dmatrix = xgb.DMatrix(data[:, 2:], label=data[:, 0], weight=data[:, 1])
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
        examples = []

        pipes_path = pipe_path if isinstance(pipe_path, list) else [pipe_path]
        for path in pipes_path:
            f = mlio.SageMakerPipe(path)
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
            pipes_path = path if isinstance(path, list) else [path]
            dataset = [mlio.SageMakerPipe(pipe_path) for pipe_path in pipes_path]
        else:
            dataset = mlio.list_files(path)

        reader_params = mlio.DataReaderParams(dataset=dataset, batch_size=BATCH_SIZE)
        reader = mlio.RecordIOProtobufReader(reader_params)

        if reader.peek_example() is not None:
            # recordio-protobuf tensor may be dense (use numpy) or sparse (use scipy)
            is_dense_tensor = type(reader.peek_example()["values"]) is mlio.DenseTensor

            all_features = []
            all_labels = []
            for example in reader:
                features = as_numpy(example["values"]) if is_dense_tensor else to_coo_matrix(example["values"])
                all_features.append(features)

                labels = as_numpy(example["label_values"])
                all_labels.append(labels)

            all_features = np.vstack(all_features) if is_dense_tensor else scipy_vstack(all_features).tocsr()
            all_labels = np.concatenate(all_labels, axis=None)
            dmatrix = xgb.DMatrix(all_features, label=all_labels)
            return dmatrix
        else:
            return None

    except Exception as e:
        raise exc.UserError("Failed to load recordio-protobuf data with exception:\n{}".format(e))


def _get_pipe_mode_files_path(data_path: Union[List[str], str]) -> List[str]:
    """
    :param data_path: Either directory or file
    """
    if isinstance(data_path, list):
        files_path = data_path
    else:
        files_path = [data_path]
        if not os.path.exists(f"{data_path}_0"):
            logging.info(f"Pipe path {data_path} does not exist!")
            return None
    return files_path


def _make_symlinks_from_a_folder(dest_path: str, data_path: str, depth: int):
    if (depth > MAX_FOLDER_DEPTH):
        raise exc.UserError(f"Folder depth exceed the limit: {MAX_FOLDER_DEPTH}.")

    if os.path.isfile(data_path):
        _make_symlink(data_path, dest_path, os.path.basename(data_path))
        return
    else:
        logging.info(f"Making smlinks from folder {data_path} to folder {dest_path}")
        for item in os.scandir(data_path):
            if item.is_file():
                _make_symlink(item.path, dest_path, item.name)
            elif item.is_dir():
                _make_symlinks_from_a_folder(dest_path, item.path, depth + 1)


def _make_symlinks_from_a_folder_with_warning(dest_path: str, data_path: str):
    """
    :param dest_path: A dir
    :param data_path: Either dir or file
    :param depth: current folder depth, Integer
    """

    # If data_path is a single file A, create smylink A -> dest_path/A
    # If data_path is a dir, create symlinks for files located within depth of MAX_FOLDER_DEPTH
    # under this dir. Ignore the files in deeper sub dirs and log a warning if they exist.

    if (not os.path.exists(dest_path)) or (not os.path.exists(data_path)):
        raise exc.AlgorithmError(f"Unable to create symlinks as {data_path} or {dest_path} doesn't exist ")

    if (not os.path.isdir(dest_path)):
        raise exc.AlgorithmError(f"Unable to create symlinks as dest_path {dest_path} is not a dir")

    try:
        _make_symlinks_from_a_folder(dest_path, data_path, 1)
    except exc.UserError as e:
        if e.message == f"Folder depth exceed the limit: {MAX_FOLDER_DEPTH}.":
            logging.warning(
                f"The depth of folder {data_path} exceed the limit {MAX_FOLDER_DEPTH}."
                f" Files in deeper sub dirs won't be loaded."
                f" Please adjust the folder structure accordingly."
                )


def _get_file_mode_files_path(data_path: Union[List[str], str]) -> List[str]:
    """
    :param data_path: Either directory or file
    """
    # In file mode, we create a temp directory with symlink to all input files or
    # directories to meet XGB's assumption that all files are in the same directory.

    logging.info("File path {} of input files".format(data_path))
    # Create a directory with symlinks to input files.
    files_path = "/tmp/sagemaker_xgboost_input_data"
    shutil.rmtree(files_path, ignore_errors=True)
    os.mkdir(files_path)
    if isinstance(data_path, list):
        for path in data_path:
            _make_symlinks_from_a_folder_with_warning(files_path, path)
    else:
        if not os.path.exists(data_path):
            logging.info("File path {} does not exist!".format(data_path))
            return None
        elif os.path.isdir(data_path) or os.path.isfile(data_path):
            # traverse all sub-dirs to load all training data
            _make_symlinks_from_a_folder_with_warning(files_path, data_path)
        else:
            exc.UserError("Unknown input files path: {}".format(data_path))

    return files_path


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

    # To get best results from cross validation, we should merge train_dmatrix
    # and val_dmatrix for bigger data. However, DMatrix doesn't support concat
    # operation and it cannot be exported to other formats (e.g. numpy).
    # It is possible to write it to a file in binary format matrix.save("data.buffer").
    # However, xgb doesn't support read multiple buffer files.
    #
    # So the only way to combine the data is to read them in one shot.
    # Fortunately, milo can read multiple pipes together. So we extends
    # the parameter data_path to support list. If data_path is string as usual,
    # get_dmatrix will work as before. When it is a list, it works as explained in respective functions.

    if is_pipe:
        files_path = _get_pipe_mode_files_path(data_path)
    else:
        files_path = _get_file_mode_files_path(data_path)
    logging.info(f"files path: {files_path}")
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
            "is less than number of workers (16 * number of instance) in the cluster."
        )

    return dmatrix


def get_size(data_path, is_pipe=False):
    """Return size of data files at dir_path.

    :param data_path: Either directory or file
    :param is_pipe: Boolean to indicate if data is being read in pipe mode
    :return: Size of data or 1 if sagemaker pipe found
    """
    if is_pipe and os.path.exists(f"{data_path}_0"):
        logging.info(f"Pipe path {data_path} found.")
        return 1
    if not os.path.exists(data_path):
        logging.info(f"Path {data_path} does not exist!")
        return 0
    else:
        total_size = 0
        if os.path.isfile(data_path):
            return os.path.getsize(data_path)
        else:
            for root, dirs, files in os.walk(data_path):
                for current_file in files:
                    if current_file.startswith("."):
                        raise exc.UserError("Hidden file found in the data path! Remove that before training.")
                    file_path = os.path.join(root, current_file)
                    total_size += os.path.getsize(file_path)
            return total_size


def _make_symlink(path, source_path, name):
    base_name = os.path.join(source_path, name)
    file_name = base_name + str(hash(path))
    logging.info(f"creating symlink between Path {path} and destination {file_name}")
    os.symlink(path, file_name)


def check_data_redundancy(train_path, validate_path):
    """Log a warning if suspected duplicate files are found in the training and validation folders.

    The validation score of models would be invalid if the same data is used for both training and validation.
    Files are suspected of being duplicates when the file names are the same and their sizes are the same.

    param train_path : path to training data
    param validate_path : path to validation data
    """
    if not os.path.exists(train_path):
        raise exc.UserError("training data's path is not existed")
    if not os.path.exists(validate_path):
        raise exc.UserError("validation data's path is not existed")

    training_files_set = set(f for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f)))
    validation_files_set = set(f for f in os.listdir(validate_path) if os.path.isfile(os.path.join(validate_path, f)))
    same_name_files = training_files_set.intersection(validation_files_set)
    for f in same_name_files:
        f_train_path = os.path.join(train_path, f)
        f_validate_path = os.path.join(validate_path, f)
        f_train_size = os.path.getsize(f_train_path)
        f_validate_size = os.path.getsize(f_validate_path)
        if f_train_size == f_validate_size:
            logging.warning(
                f"Suspected identical files found. ({f_train_path} and {f_validate_path}"
                f"with same size {f_validate_size} bytes)."
                f" Note: Duplicate data in the training set and validation set is usually"
                f" not intentional and can impair the validity of the model evaluation by"
                f" the validation score."
            )
