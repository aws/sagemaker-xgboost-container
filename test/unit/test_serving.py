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
from mock import MagicMock, patch
import numpy as np
import os
import pytest
import xgboost as xgb

from sagemaker_algorithm_toolkit.exceptions import UserError
from sagemaker_containers.beta.framework import (content_types, encoders, errors)
from sagemaker_xgboost_container import serving
from sagemaker_xgboost_container.constants import sm_env_constants

TEST_CONFIG_FILE = "test_dir"


@pytest.fixture(autouse=True)
def mock_set_mms_config_file(monkeypatch):
    monkeypatch.setenv('XGBOOST_MMS_CONFIG', TEST_CONFIG_FILE)


@pytest.fixture(scope='module', name='np_array')
def fixture_np_array():
    return np.ones((2, 2))


class FakeEstimator:
    def __init__(self):
        pass

    @staticmethod
    def predict(input):
        return


@pytest.mark.parametrize('csv_array', ('42,6,9', '42.0,6.0,9.0'))
def test_input_fn_dmatrix(csv_array):
    deserialized_csv_array = serving.default_input_fn(csv_array, content_types.CSV)
    assert type(deserialized_csv_array) is xgb.DMatrix


def test_input_fn_bad_content_type():
    with pytest.raises(errors.UnsupportedFormatError):
        serving.default_input_fn('', 'application/not_supported')


def test_default_model_fn():
    with pytest.raises(NotImplementedError):
        serving.default_model_fn('model_dir')


def test_predict_fn(np_array):
    mock_estimator = FakeEstimator()
    with patch.object(mock_estimator, 'predict') as mock:
        serving.default_predict_fn(np_array, mock_estimator)
    mock.assert_called_once()


def test_output_fn_json(np_array):
    response = serving.default_output_fn(np_array, content_types.JSON)

    assert response.get_data(as_text=True) == encoders.array_to_json(np_array.tolist())
    assert response.content_type == content_types.JSON


def test_output_fn_csv(np_array):
    response = serving.default_output_fn(np_array, content_types.CSV)

    assert response.get_data(as_text=True) == '1.0,1.0\n1.0,1.0\n'
    # TODO This is a workaround to get the test passsing.
    # Not sure if it is related to executing tests on Mac in specific virtual environment,
    # but the content type in response is: 'text/csv; charset=utf-8' instead of the expected: text/csv
    assert content_types.CSV in response.content_type


def test_output_fn_npz(np_array):
    response = serving.default_output_fn(np_array, content_types.NPY)

    assert response.get_data() == encoders.array_to_npy(np_array)
    assert response.content_type == content_types.NPY


def test_input_fn_bad_accept():
    with pytest.raises(errors.UnsupportedFormatError):
        serving.default_output_fn('', 'application/not_supported')


@patch('sagemaker_xgboost_container.serving.server')
def test_serving_entrypoint_start_gunicorn(mock_server):
    mock_server.start = MagicMock()
    serving.serving_entrypoint()
    mock_server.start.assert_called_once()


@patch('sagemaker_xgboost_container.serving.server')
@patch('sagemaker_xgboost_container.serving.set_default_serving_env_if_unspecified')
def test_serving_entrypoint_set_default_env(mock_set_default_serving_env_if_unspecified, mock_server):
    serving.serving_entrypoint()
    mock_set_default_serving_env_if_unspecified.assert_called_once()
    assert os.getenv('OMP_NUM_THREADS') == sm_env_constants.ONE_THREAD_PER_PROCESS
    with patch.dict(os.environ, {"OMP_NUM_THREADS": "USER_SPECIFIED_VALUE"}, clear=True):
        mock_set_default_serving_env_if_unspecified.reset_mock()
        serving.serving_entrypoint()
        mock_set_default_serving_env_if_unspecified.assert_called_once()
        assert os.getenv('OMP_NUM_THREADS') == "USER_SPECIFIED_VALUE"


@patch.dict(os.environ, {'SAGEMAKER_MULTI_MODEL': 'True', })
@patch('sagemaker_xgboost_container.serving.start_mxnet_model_server')
def test_serving_entrypoint_start_mms(mock_start_mxnet_model_server):
    serving.serving_entrypoint()
    mock_start_mxnet_model_server.assert_called_once()


@patch('sagemaker_xgboost_container.serving.transformer')
def test_user_module_transformer_with_transform_and_other_fn(mock_transformer):
    mock_module = MagicMock(spec=["model_fn", "transform_fn", "input_fn"])
    with pytest.raises(UserError):
        serving._user_module_transformer(mock_module)


@patch('sagemaker_xgboost_container.serving.transformer')
def test_user_module_transformer_with_transform_and_no_other_fn(mock_transformer):
    mock_module = MagicMock(spec=["model_fn", "transform_fn"])
    serving._user_module_transformer(mock_module)
    mock_transformer.Transformer.assert_called_once_with(
        model_fn=mock_module.model_fn,
        transform_fn=mock_module.transform_fn
    )


@patch('sagemaker_xgboost_container.serving.transformer')
def test_user_module_transformer_with_model_fn_only(mock_transformer):
    mock_module = MagicMock(spec=["model_fn"])
    serving._user_module_transformer(mock_module)
    mock_transformer.Transformer.assert_called_once_with(
        model_fn=mock_module.model_fn,
        input_fn=serving.default_input_fn,
        predict_fn=serving.default_predict_fn,
        output_fn=serving.default_output_fn
    )


@patch('sagemaker_xgboost_container.serving.transformer')
def test_user_module_transformer_with_input_fn(mock_transformer):
    mock_module = MagicMock(spec=["model_fn", "input_fn"])
    serving._user_module_transformer(mock_module)
    mock_transformer.Transformer.assert_called_once_with(
        model_fn=mock_module.model_fn,
        input_fn=mock_module.input_fn,
        predict_fn=serving.default_predict_fn,
        output_fn=serving.default_output_fn
    )


@patch('sagemaker_xgboost_container.serving.transformer')
def test_user_module_transformer_with_predict_fn(mock_transformer):
    mock_module = MagicMock(spec=["model_fn", "predict_fn"])
    serving._user_module_transformer(mock_module)
    mock_transformer.Transformer.assert_called_once_with(
        model_fn=mock_module.model_fn,
        input_fn=serving.default_input_fn,
        predict_fn=mock_module.predict_fn,
        output_fn=serving.default_output_fn
    )


@patch('sagemaker_xgboost_container.serving.transformer')
def test_user_module_transformer_with_output_fn(mock_transformer):
    mock_module = MagicMock(spec=["model_fn", "output_fn"])
    serving._user_module_transformer(mock_module)
    mock_transformer.Transformer.assert_called_once_with(
        model_fn=mock_module.model_fn,
        input_fn=serving.default_input_fn,
        predict_fn=serving.default_predict_fn,
        output_fn=mock_module.output_fn
    )
