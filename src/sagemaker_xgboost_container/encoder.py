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
from __future__ import absolute_import

import cgi
import csv
import io
import json
import logging
import os
import tempfile
from typing import Iterable, Union

import mlio
from mlio.integ.numpy import as_numpy
from mlio.integ.scipy import to_coo_matrix
import numpy as np
from sagemaker_containers import _content_types, _errors
from scipy.sparse import vstack as scipy_vstack
import xgboost as xgb

from sagemaker_xgboost_container.constants import xgb_content_types


def _clean_csv_string(csv_string, delimiter):
    return ['nan' if x == '' else x for x in csv_string.split(delimiter)]


def csv_to_dmatrix(input: Union[str, bytes], dtype=None) -> xgb.DMatrix:
    """Convert a CSV object to a DMatrix object.
    Args:
        input (str/binary): CSV string or binary object. Assumes the string has been stripped of leading or trailing newline chars.
        dtype (dtype, optional):  Data type of the resulting array. If None, the dtypes will be determined by the
                                        contents of each column, individually. This argument can only be used to
                                        'upcast' the array.  For downcasting, use the .astype(t) method.
    Returns:
        (xgb.DMatrix): XGBoost DataMatrix
    """
    if isinstance(input, bytes):
        csv_string = input.decode()
    else:
        csv_string = input
    sniff_delimiter = csv.Sniffer().sniff(csv_string.split('\n')[0][:512]).delimiter
    delimiter = ',' if sniff_delimiter.isalnum() else sniff_delimiter
    logging.info("Determined delimiter of CSV input is \'{}\'".format(delimiter))

    np_payload = np.array(list(map(lambda x: _clean_csv_string(x, delimiter), csv_string.split('\n')))).astype(dtype)
    return xgb.DMatrix(np_payload)


def libsvm_to_dmatrix(string_like):  # type: (bytes) -> xgb.DMatrix
    """Convert a LIBSVM string representation to a DMatrix object.
    Args:
        string_like (bytes): LIBSVM string.
    Returns:
        (xgb.DMatrix): XGBoost DataMatrix
    """
    temp_file_location = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as libsvm_file:
            temp_file_location = libsvm_file.name
            libsvm_file.write(string_like)

        dmatrix = xgb.DMatrix(temp_file_location)
    finally:
        if temp_file_location and os.path.exists(temp_file_location):
            os.remove(temp_file_location)

    return dmatrix


def recordio_protobuf_to_dmatrix(string_like):  # type: (bytes) -> xgb.DMatrix
    """Convert a RecordIO-Protobuf byte representation to a DMatrix object.
    Args:
        string_like (bytes): RecordIO-Protobuf bytes.
    Returns:
    (xgb.DMatrix): XGBoost DataMatrix
    """
    buf = bytes(string_like)
    dataset = [mlio.InMemoryStore(buf)]
    reader_params = mlio.DataReaderParams(dataset=dataset, batch_size=100)
    reader = mlio.RecordIOProtobufReader(reader_params)

    is_dense_tensor = type(reader.peek_example()['values']) is mlio.DenseTensor

    examples = []
    for example in reader:
        # Ignore labels if present
        values = as_numpy(example['values']) if is_dense_tensor else to_coo_matrix(example['values'])
        examples.append(values)

    data = np.vstack(examples) if is_dense_tensor else scipy_vstack(examples).tocsr()
    dmatrix = xgb.DMatrix(data)
    return dmatrix


_dmatrix_decoders_map = {
    _content_types.CSV: csv_to_dmatrix,
    xgb_content_types.LIBSVM: libsvm_to_dmatrix,
    xgb_content_types.X_LIBSVM: libsvm_to_dmatrix,
    xgb_content_types.X_RECORDIO_PROTOBUF: recordio_protobuf_to_dmatrix}


def json_to_jsonlines(json_data):
    """Convert a json response to jsonlines.

    :param json_data: json data (dict or json string)
    :return: jsonlines encoded response (bytes)
    """
    resp_dict = json_data if isinstance(json_data, dict) else json.loads(json_data)

    if len(resp_dict.keys()) != 1:
        raise ValueError("JSON response is not compatible for conversion to jsonlines.")

    bio = io.BytesIO()
    for value in resp_dict.values():
        for entry in value:
            bio.write(bytes(json.dumps(entry) + "\n", "UTF-8"))
    return bio.getvalue()


def decode(obj, content_type):
    # type: (np.array or Iterable or int or float, str) -> xgb.DMatrix
    """Decode an object ton a one of the default content types to a DMatrix object.
    Args:
        obj (object): to be decoded.
        content_type (str): content type to be used.
    Returns:
        np.array: decoded object.
    """
    try:
        media_content_type, _params = cgi.parse_header(content_type)
        decoder = _dmatrix_decoders_map[media_content_type]
        return decoder(obj)
    except KeyError:
        raise _errors.UnsupportedFormatError(media_content_type)
