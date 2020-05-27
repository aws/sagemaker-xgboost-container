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

from mock import MagicMock
import os
import pytest

from sagemaker_algorithm_toolkit import exceptions as exc
from sagemaker_xgboost_container import data_utils
from sagemaker_xgboost_container.data_utils import CSV, LIBSVM, RECORDIO_PROTOBUF
from sagemaker_xgboost_container.algorithm_mode import serve_utils


TEST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESOURCES_PATH = os.path.join(TEST_DIR, 'resources')


@pytest.mark.parametrize('csv_content_type', ('csv', 'text/csv', 'text/csv; label_size=1',
                                              'text/csv;label_size = 1', 'text/csv; charset=utf-8',
                                              'text/csv; label_size=1; charset=utf-8'))
def test_parse_csv_data(csv_content_type):
    data_payload = b'1,1'
    parsed_payload, parsed_content_type = serve_utils.parse_content_data(data_payload, csv_content_type)
    assert parsed_content_type == data_utils.CSV


@pytest.mark.parametrize('libsvm_content_type', ('libsvm', 'text/libsvm', 'text/x-libsvm'))
def test_parse_libsvm_data(libsvm_content_type):
    data_payload = b'0:1'
    parsed_payload, parsed_content_type = serve_utils.parse_content_data(data_payload, libsvm_content_type)
    assert parsed_content_type == data_utils.LIBSVM


@pytest.mark.parametrize('incorrect_content_type', ('incorrect_format', 'text/csv; label_size=5',
                                                    'text/csv; label_size=1=1', 'text/csv; label_size=1; label_size=2',
                                                    'label_size=1; text/csv'))
def test_incorrect_content_type(incorrect_content_type):
    data_payload = '0'
    with pytest.raises(exc.UserError):
        serve_utils.parse_content_data(data_payload, incorrect_content_type)


@pytest.mark.parametrize('model_info', (('pickled_model', serve_utils.PKL_FORMAT),
                                        ('saved_booster', serve_utils.XGB_FORMAT)))
def test_get_loaded_booster(model_info):
    """Test model loading

    'pickled_model' directory has a model dumped using pickle module
    'saved_booster' directory has a model saved using booster.save_model()
    """
    model_dir_name, model_format = model_info
    model_dir = os.path.join(RESOURCES_PATH, 'models', model_dir_name)
    loaded_booster, loaded_model_format = serve_utils.get_loaded_booster(model_dir)
    assert loaded_model_format == model_format


@pytest.mark.parametrize('correct_content_type', (CSV, LIBSVM, RECORDIO_PROTOBUF))
def test_predict_valid_content_type(correct_content_type):
    mock_feature_names = [0, 1, 2, 3]

    mock_booster = MagicMock()
    mock_booster.predict = MagicMock()
    mock_booster.feature_names = mock_feature_names
    mock_dmatrix = MagicMock()
    mock_dmatrix.feature_names = mock_feature_names

    serve_utils.predict(mock_booster, serve_utils.PKL_FORMAT, mock_dmatrix, correct_content_type)
