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
import json
from mock import Mock, patch
import mock
import os
from pathlib import Path
import pytest
import tempfile

from sagemaker_containers import _content_types, _errors
import xgboost as xgb

from sagemaker_xgboost_container import encoder


@pytest.mark.parametrize('target', ('42,6,9', '42.0,6.0,9.0', '42\n6\n9\n'))
def test_csv_to_dmatrix(target):
    actual = encoder.csv_to_dmatrix(target)
    assert type(actual) is xgb.DMatrix


@pytest.mark.parametrize(
    'target', ('1,2,3,12:12:12',
               '1,2,3,2019-1-1',
               '1,2,3,2019-1-1 12:12:12',
               '1,2,3,2019-1-1 12:12:12+00',
               '1,2,3,-14 days',
               '1,2,3\n1,2,c'))
def test_csv_to_dmatrix_error(target):
    try:
        encoder.csv_to_dmatrix(target)
        assert False
    except Exception as e:
        assert type(e) is ValueError


@pytest.mark.parametrize('target', (b'0 0:1 5:1', b'0:1 5:1'))
def test_libsvm_to_dmatrix(target):
    temp_libsvm_file = tempfile.NamedTemporaryFile(delete=False)
    temp_libsvm_file_name = temp_libsvm_file.name
    assert os.path.exists(temp_libsvm_file_name)

    with mock.patch('sagemaker_xgboost_container.encoder.tempfile') as mock_tempfile:
        mock_tempfile.NamedTemporaryFile.return_value = temp_libsvm_file
        actual = encoder.libsvm_to_dmatrix(target)

    assert type(actual) is xgb.DMatrix
    assert not os.path.exists(temp_libsvm_file_name)


@pytest.mark.parametrize(
    'target', (b'\n#\xd7\xce\x13\x00\x00\x00\n\x11\n\x06values\x12\x07:\x05\n\x03*\x06\t\x00',  # 42,6,9
               b'\n#\xd7\xce(\x00\x00\x00\n&\n\x06values\x12\x1c\x1a\x1a\n\x18\x00\x00\x00'  # 42.0,6.0,9.0
               b'\x00\x00\x00E@\x00\x00\x00\x00\x00\x00\x18@\x00\x00\x00\x00\x00\x00"@',
               b'\n#\xd7\xce\x19\x00\x00\x00\n\x17\n\x06values\x12\r:\x0b\n\x02\x01\x01\x12'  # 0:1 5:1
               b'\x02\x00\x05\x1a\x01\x06\x00\x00\x00'))
def test_recordio_protobuf_to_dmatrix(target):
    actual = encoder.recordio_protobuf_to_dmatrix(target)
    assert type(actual) is xgb.DMatrix


def test_sparse_recordio_protobuf_to_dmatrix():
    current_path = Path(os.path.abspath(__file__))
    data_path = os.path.join(str(current_path.parent.parent), 'resources', 'data')
    files_path = os.path.join(data_path, 'recordio_protobuf', 'sparse_edge_cases')

    for filename in os.listdir(files_path):
        file_path = os.path.join(files_path, filename)
        with open(file_path, 'rb') as f:
            target = f.read()
            actual = encoder.recordio_protobuf_to_dmatrix(target)
            assert type(actual) is xgb.DMatrix


def test_decode_error():
    with pytest.raises(_errors.UnsupportedFormatError):
        encoder.decode(42, _content_types.OCTET_STREAM)


@pytest.mark.parametrize('content_type', [_content_types.JSON, _content_types.CSV])
def test_decode(content_type):
    decoder = Mock()
    with patch.dict(encoder._dmatrix_decoders_map, {content_type: decoder}, clear=True):
        encoder.decode(42, content_type)

    decoder.assert_called_once_with(42)


@pytest.mark.parametrize('content_type', ['text/csv; charset=UTF-8'])
def test_decode_with_complex_csv_content_type(content_type):
    dmatrix_result = encoder.decode("42.0,6.0,9.0\n42.0,6.0,9.0", content_type)
    assert type(dmatrix_result) is xgb.DMatrix


def test_encoder_jsonlines_from_json():
    json_response = json.dumps({'predictions': [{"predicted_label": 1, "probabilities": [0.4, 0.6]},
                                                {"predicted_label": 0, "probabilities": [0.9, 0.1]}]})
    expected_jsonlines = b'{"predicted_label": 1, "probabilities": [0.4, 0.6]}\n' \
                         b'{"predicted_label": 0, "probabilities": [0.9, 0.1]}\n'

    jsonlines_response = encoder.json_to_jsonlines(json_response)
    assert expected_jsonlines == jsonlines_response


def test_encoder_jsonlines_from_json_error():
    bad_json_response = json.dumps({'predictions': [], 'metadata': []})
    with pytest.raises(ValueError):
        encoder.json_to_jsonlines(bad_json_response)
