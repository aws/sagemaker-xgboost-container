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
import pytest
import xgboost as xgb

from sagemaker_containers.beta.framework import (content_types, encoders, errors)
from sagemaker_xgboost_container.handler_service import HandlerService


handler = HandlerService().DefaultXGBoostUserModuleInferenceHandler()


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
    deserialized_csv_array = handler.default_input_fn(csv_array, content_types.CSV)
    assert type(deserialized_csv_array) is xgb.DMatrix


def test_input_fn_bad_content_type():
    with pytest.raises(errors.UnsupportedFormatError):
        handler.default_input_fn('', 'application/not_supported')


def test_default_model_fn():
    with pytest.raises(NotImplementedError):
        handler.default_model_fn('model_dir')


def test_predict_fn(np_array):
    mock_estimator = FakeEstimator()
    with patch.object(mock_estimator, 'predict') as mock:
        handler.default_predict_fn(np_array, mock_estimator)
    mock.assert_called_once()


def test_output_fn_json(np_array):
    response = handler.default_output_fn(np_array, content_types.JSON)
    assert response == encoders.array_to_json(np_array.tolist())


def test_output_fn_csv(np_array):
    response = handler.default_output_fn(np_array, content_types.CSV)
    assert response == b'1.0,1.0\n1.0,1.0\n'


def test_output_fn_npz(np_array):
    response = handler.default_output_fn(np_array, content_types.NPY)
    assert response == encoders.array_to_npy(np_array)


def test_input_fn_bad_accept():
    with pytest.raises(errors.UnsupportedFormatError):
        handler.default_output_fn('', 'application/not_supported')
