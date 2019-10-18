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
from mock import patch

from sagemaker_xgboost_container import serving


@patch('sagemaker_inference.model_server.start_model_server')
def test_hosting_algorithm_mode(start_model_server):
    serving.main()
    start_model_server.assert_called_with(
        handler_service='sagemaker_xgboost_container.algorithm_mode.handler_service')


@patch('sagemaker_inference.model_server.start_model_server')
@patch('sagemaker_xgboost_container.serving.env.ServingEnv.module_name', return_value='foo')
def test_hosting_user_mode(user_module_name, start_model_server):
    serving.main()
    start_model_server.assert_called_with(
        handler_service='sagemaker_xgboost_container.handler_service')
