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
from mock import Mock, patch
import mock
import os
import pytest
import tempfile

import xgboost as xgb

from sagemaker_containers import _content_types, _errors
from sagemaker_xgboost_container import encoder


@pytest.mark.parametrize('target', ('42,6,9', '42.0,6.0,9.0', '42\n6\n9\n'))
def test_csv_to_dmatrix(target):
    actual = encoder.csv_to_dmatrix(target)
    assert type(actual) is xgb.DMatrix


@pytest.mark.parametrize('target', ('0 0:1 5:1', '0:1 5:1'))
def test_libsvm_to_dmatrix(target):
    temp_libsvm_file = tempfile.NamedTemporaryFile(delete=False)
    temp_libsvm_file_name = temp_libsvm_file.name
    assert os.path.exists(temp_libsvm_file_name)

    with mock.patch('sagemaker_xgboost_framework.encoder.tempfile') as mock_tempfile:
        mock_tempfile.NamedTemporaryFile.return_value = temp_libsvm_file
        actual = encoder.libsvm_to_dmatrix(target)

    assert type(actual) is xgb.DMatrix
    assert not os.path.exists(temp_libsvm_file_name)


def test_decode_error():
    with pytest.raises(_errors.UnsupportedFormatError):
        encoder.decode(42, _content_types.OCTET_STREAM)


@pytest.mark.parametrize('content_type', [_content_types.JSON, _content_types.CSV])
def test_decode(content_type):
    decoder = Mock()
    with patch.dict(encoder._dmatrix_decoders_map, {content_type: decoder}, clear=True):
        encoder.decode(42, content_type)

    decoder.assert_called_once_with(42)
