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

import json
from mock import MagicMock, patch
import os
import pytest

from sagemaker_algorithm_toolkit import exceptions as exc
from sagemaker_xgboost_container import data_utils
from sagemaker_xgboost_container.algorithm_mode import serve


@pytest.mark.parametrize('csv_content_type', ('csv', 'text/csv', 'text/csv; label_size=1',
                                              'text/csv;label_size = 1', 'text/csv; charset=utf-8',
                                              'text/csv; label_size=1; charset=utf-8'))
def test_parse_csv_data(csv_content_type):
    data_payload = b'1,1'
    mock_request = MagicMock()
    mock_request.data = data_payload
    mock_request.content_type = csv_content_type

    parsed_payload, parsed_content_type = serve._parse_content_data(mock_request)

    assert parsed_content_type == data_utils.CSV


@pytest.mark.parametrize('libsvm_content_type', ('libsvm', 'text/libsvm', 'text/x-libsvm'))
def test_parse_libsvm_data(libsvm_content_type):
    data_payload = b'0:1'
    mock_request = MagicMock()
    mock_request.data = data_payload
    mock_request.content_type = libsvm_content_type

    parsed_payload, parsed_content_type = serve._parse_content_data(mock_request)

    assert parsed_content_type == data_utils.LIBSVM


@pytest.mark.parametrize('incorrect_content_type', ('incorrect_format', 'text/csv; label_size=5',
                                                    'text/csv; label_size=1=1', 'text/csv; label_size=1; label_size=2',
                                                    'label_size=1; text/csv'))
def test_incorrect_content_type(incorrect_content_type):
    data_payload = '0'
    mock_request = MagicMock()
    mock_request.data = data_payload
    mock_request.content_type = incorrect_content_type

    with pytest.raises(exc.UserError):
        serve._parse_content_data(mock_request)


def test_default_execution_parameters():
    execution_parameters_response = serve.execution_parameters()

    parsed_exec_params_response = json.loads(execution_parameters_response.response[0])
    assert parsed_exec_params_response['MaxPayloadInMB'] == 6
    assert parsed_exec_params_response["BatchStrategy"] == "MULTI_RECORD"


@patch.dict(os.environ, {"MAX_CONTENT_LENGTH": '21'})
def test_max_execution_parameters():
    execution_parameters_response = serve.execution_parameters()

    parsed_exec_params_response = json.loads(execution_parameters_response.response[0])
    assert parsed_exec_params_response['MaxPayloadInMB'] == 20
    assert parsed_exec_params_response["BatchStrategy"] == "MULTI_RECORD"
