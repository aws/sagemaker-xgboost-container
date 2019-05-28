# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import csv
import logging
import os
import tempfile
from typing import Iterable

import numpy as np
import xgboost as xgb

from sagemaker_containers import _content_types, _errors
from sagemaker_xgboost_container.constants import xgb_content_types


def _clean_csv_string(csv_string, delimiter):
    return ['nan' if x == '' else x for x in csv_string.split(delimiter)]


def csv_to_dmatrix(string_like, dtype=None):  # type: (str) -> xgb.DMatrix
    """Convert a CSV object to a DMatrix object.
    Args:
        string_like (str): CSV string.
        dtype (dtype, optional):  Data type of the resulting array. If None, the dtypes will be determined by the
                                        contents of each column, individually. This argument can only be used to
                                        'upcast' the array.  For downcasting, use the .astype(t) method.
    Returns:
        (xgb.DMatrix): XGBoost DataMatrix
    """
    sniff_delimiter = csv.Sniffer().sniff(string_like.split('\n')[0][:512]).delimiter
    delimiter = ',' if sniff_delimiter.isalnum() else sniff_delimiter
    logging.info("Determined delimiter of CSV input is \'{}\'".format(delimiter))

    np_payload = np.array(list(map(lambda x: _clean_csv_string(x, delimiter), string_like.split('\n')))).astype(dtype)
    return xgb.DMatrix(np_payload)


def libsvm_to_dmatrix(string_like):  # type: (str) -> xgb.DMatrix
    """Convert a LIBSVM string representation to a DMatrix object.
    Args:
        string_like (str): LIBSVM string.
    Returns:
        (xgb.DMatrix): XGBoost DataMatrix
    """
    temp_file_location = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as libsvm_file:
            temp_file_location = libsvm_file.name
            libsvm_file.write(string_like.encode())

        dmatrix = xgb.DMatrix(temp_file_location)
    finally:
        if temp_file_location and os.path.exists(temp_file_location):
            os.remove(temp_file_location)

    return dmatrix


_dmatrix_decoders_map = {
    _content_types.CSV: csv_to_dmatrix,
    xgb_content_types.LIBSVM: libsvm_to_dmatrix,
    xgb_content_types.X_LIBSVM: libsvm_to_dmatrix}


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
        decoder = _dmatrix_decoders_map[content_type]
        return decoder(obj)
    except KeyError:
        raise _errors.UnsupportedFormatError(content_type)
