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
from mock import patch

from sagemaker_xgboost_container.algorithm_mode import serve


def test_default_execution_parameters():
    execution_parameters_response = serve.execution_parameters()

    parsed_exec_params_response = json.loads(execution_parameters_response.response[0])
    assert parsed_exec_params_response['MaxPayloadInMB'] == 6
    assert parsed_exec_params_response["BatchStrategy"] == "MULTI_RECORD"


@patch('sagemaker_xgboost_container.algorithm_mode.serve.PARSED_MAX_CONTENT_LENGTH', 19 * 1024 ** 2)
def test_max_execution_parameters():
    execution_parameters_response = serve.execution_parameters()

    parsed_exec_params_response = json.loads(execution_parameters_response.response[0])
    assert parsed_exec_params_response['MaxPayloadInMB'] == 19
    assert parsed_exec_params_response["BatchStrategy"] == "MULTI_RECORD"
