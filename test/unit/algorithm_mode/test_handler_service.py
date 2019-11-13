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
import numpy as np
import os
import pytest
import xgboost as xgb

from sagemaker_containers.beta.framework import (content_types, encoders)
from sagemaker_xgboost_container.algorithm_mode.handler_service import HandlerService
from sagemaker_xgboost_container.algorithm_mode import inference_errors as exc


handler = HandlerService().DefaultXGBoostAlgoModeInferenceHandler()

path = path = os.path.abspath(__file__)
resource_path = os.path.join(os.path.dirname(path), '..', '..', 'resources')


@pytest.fixture(scope='module', name='np_array')
def fixture_np_array():
    return np.ones((2, 2))


class FakeEstimator:
    def __init__(self):
        pass

    @staticmethod
    def predict(input):
        return


class FakeData:
    def __init__(self, array):
        self.feature_names = array


@pytest.mark.parametrize('csv_array', (b'42,6,9', b'42.0,6.0,9.0'))
def test_input_fn_dmatrix(csv_array):
    deserialized_csv_array, content_type = handler.default_input_fn(csv_array, content_types.CSV)
    assert type(deserialized_csv_array) is xgb.DMatrix

    deserialized_csv_array, content_type = handler.default_input_fn(csv_array, 'text/CSV;charset=utf-8')
    assert type(deserialized_csv_array) is xgb.DMatrix


def test_input_fn_bad_content_type():
    with pytest.raises(exc.NoContentInferenceError):
        handler.default_input_fn('', 'application/not_supported')


@pytest.mark.parametrize('model_dir', ['sanity', 'boston'])
def test_default_model_fn(model_dir):
    booster, format = handler.default_model_fn('{}/models/{}'.format(resource_path, model_dir))
    assert type(booster) is xgb.Booster
    assert format == 'pkl_format'


def test_predict_fn(np_array):
    mock_estimator = FakeEstimator()
    with patch.object(mock_estimator, 'predict') as mock:
        handler.default_predict_fn((np_array, 'foo_content_type'), (mock_estimator, 'foo_format'))
    mock.assert_called_once()

    mock_estimator2 = FakeEstimator()
    mock_estimator2.feature_names = np.ones(2)
    mock_data = FakeData(np.ones(2))
    with patch.object(mock_estimator2, 'predict') as mock2:
        handler.default_predict_fn((mock_data, 'text/CSV;charset=utf-8'), (mock_estimator2, 'pkl_format'))
    mock2.assert_called_once()


def test_output_fn_json(np_array):
    response = handler.default_output_fn(np_array, content_types.JSON)
    assert response == encoders.array_to_json(np_array.tolist())

    response = handler.default_output_fn(np_array, content_types.JSON.upper())
    assert response == encoders.array_to_json(np_array.tolist())


def test_output_fn_csv(np_array):
    response = handler.default_output_fn(np_array, content_types.CSV)
    assert response == b'[1.0, 1.0],[1.0, 1.0]'

    response = handler.default_output_fn(np_array, content_types.CSV.upper())
    assert response == b'[1.0, 1.0],[1.0, 1.0]'


def test_output_fn_bad_accept():
    with pytest.raises(exc.UnsupportedMediaTypeInferenceError):
        handler.default_output_fn('', 'application/not_supported')
