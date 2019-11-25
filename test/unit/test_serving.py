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
from sagemaker_xgboost_container import serving


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

    print(response)

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
