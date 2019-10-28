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
import pytest

from sagemaker_xgboost_container import serving
from sagemaker_xgboost_container import handler_service as user_module_handler_service
from sagemaker_xgboost_container.algorithm_mode import handler_service as algo_handler_service


TEST_CONFIG_FILE = "test_dir"
ALGO_HANDLER_SERVICE = algo_handler_service.__name__
USER_HANDLER_SERVICE = user_module_handler_service.__name__


@pytest.fixture
def mock_mms_config_file(monkeypatch):
    monkeypatch.setenv('XGBOOST_MMS_CONFIG', TEST_CONFIG_FILE)


@patch('sagemaker_xgboost_container.mms_patch.model_server.start_model_server')
def test_hosting_algorithm_mode(start_model_server, mock_mms_config_file):
    serving.main()
    start_model_server.assert_called_with(
        is_multi_model=False,
        handler_service='sagemaker_xgboost_container.algorithm_mode.handler_service',
        config_file=TEST_CONFIG_FILE)


@patch('sagemaker_xgboost_container.mms_patch.model_server.start_model_server')
@patch('sagemaker_xgboost_container.serving.env.ServingEnv.module_dir')
@patch('sagemaker_xgboost_container.serving.env.ServingEnv.module_name')
@patch('sagemaker_containers.beta.framework.modules.import_module')
def test_hosting_user_mode(import_module, user_module_name, module_dir, start_model_server, mock_mms_config_file):
    serving.main()
    start_model_server.assert_called_with(
        is_multi_model=False,
        handler_service='sagemaker_xgboost_container.handler_service',
        config_file=TEST_CONFIG_FILE)
    import_module.assert_called_with(module_dir, user_module_name)
