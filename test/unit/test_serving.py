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
import os
import pytest

from sagemaker_algorithm_toolkit import exceptions as exc
from sagemaker_xgboost_container import serving
from sagemaker_xgboost_container import handler_service as user_module_handler_service
from sagemaker_xgboost_container.algorithm_mode import handler_service as algo_handler_service


TEST_CONFIG_FILE = "test_dir"
ALGO_HANDLER_SERVICE = algo_handler_service.__name__
USER_HANDLER_SERVICE = user_module_handler_service.__name__
TEST_MAX_CONTENT_LEN = 1024
TEST_NUM_CPU = 3


@pytest.fixture
def mock_set_mms_config_file(monkeypatch):
    monkeypatch.setenv('XGBOOST_MMS_CONFIG', TEST_CONFIG_FILE)


@pytest.fixture
def mock_set_multi_model_env(monkeypatch):
    monkeypatch.setenv('SAGEMAKER_MULTI_MODEL', 'true')


@patch('sagemaker_xgboost_container.mms_patch.model_server.start_model_server')
def test_single_model_algorithm_mode_hosting(start_model_server, mock_set_mms_config_file):
    serving.main()
    start_model_server.assert_called_with(
        is_multi_model=False,
        handler_service='sagemaker_xgboost_container.algorithm_mode.handler_service',
        config_file=TEST_CONFIG_FILE)


@patch('sagemaker_xgboost_container.mms_patch.model_server.start_model_server')
def test_multi_model_algorithm_mode_hosting(start_model_server, mock_set_mms_config_file, mock_set_multi_model_env):
    serving.main()
    start_model_server.assert_called_with(
        is_multi_model=True,
        handler_service='sagemaker_xgboost_container.algorithm_mode.handler_service',
        config_file=TEST_CONFIG_FILE)


@patch('sagemaker_xgboost_container.mms_patch.model_server.start_model_server')
@patch('sagemaker_xgboost_container.serving.env.ServingEnv.module_dir')
@patch('sagemaker_xgboost_container.serving.env.ServingEnv.module_name')
@patch('sagemaker_containers.beta.framework.modules.import_module')
def test_single_model_user_mode_hosting(
        import_module,
        user_module_name,
        module_dir,
        start_model_server,
        mock_set_mms_config_file):
    serving.main()
    start_model_server.assert_called_with(
        is_multi_model=False,
        handler_service='sagemaker_xgboost_container.handler_service',
        config_file=TEST_CONFIG_FILE)
    import_module.assert_called_with(module_dir, user_module_name)


@patch('sagemaker_xgboost_container.serving.env.ServingEnv.module_name')
def test_multi_model_user_mode_hosting_error(user_module_name, mock_set_multi_model_env):
    with pytest.raises(exc.PlatformError):
        serving.main()


@patch('sagemaker_xgboost_container.mms_patch.model_server.start_model_server')
@patch('multiprocessing.cpu_count', return_value=TEST_NUM_CPU)
def test_env_var_setting_single_and_multi_model(start_model_server, mock_get_num_cpu):
    test_handler_str = 'foo'
    with patch.dict('os.environ', {}):
        serving._set_mms_configs(False, test_handler_str)

        assert os.environ["SAGEMAKER_NUM_MODEL_WORKERS"] == str(TEST_NUM_CPU)
        assert os.environ["SAGEMAKER_MMS_MODEL_STORE"] == '/opt/ml/model'
        assert os.environ["SAGEMAKER_MMS_LOAD_MODELS"] == 'ALL'
        assert os.environ["SAGEMAKER_MAX_REQUEST_SIZE"] == str(serving.DEFAULT_MAX_CONTENT_LEN)
        assert os.environ["SAGEMAKER_MMS_DEFAULT_HANDLER"] == test_handler_str

    with patch.dict('os.environ', {}):
        serving._set_mms_configs(True, test_handler_str)

        assert os.environ["SAGEMAKER_NUM_MODEL_WORKERS"] == '1'
        assert os.environ["SAGEMAKER_MMS_MODEL_STORE"] == '/'
        assert os.environ["SAGEMAKER_MMS_LOAD_MODELS"] == ''
        assert os.environ["SAGEMAKER_MAX_REQUEST_SIZE"] == str(serving.DEFAULT_MAX_CONTENT_LEN)
        assert os.environ["SAGEMAKER_MMS_DEFAULT_HANDLER"] == test_handler_str


@patch('sagemaker_xgboost_container.mms_patch.model_server.start_model_server')
def test_set_max_content_len(start_model_server):
    test_handler_str = 'foo'
    with patch.dict('os.environ', {}):
        serving._set_mms_configs(False, test_handler_str)
        assert os.environ['SAGEMAKER_MAX_REQUEST_SIZE'] == str(serving.DEFAULT_MAX_CONTENT_LEN)

    with patch.dict('os.environ', {'MAX_CONTENT_LENGTH': str(TEST_MAX_CONTENT_LEN)}):
        serving._set_mms_configs(False, test_handler_str)
        assert os.environ['SAGEMAKER_MAX_REQUEST_SIZE'] == str(TEST_MAX_CONTENT_LEN)
