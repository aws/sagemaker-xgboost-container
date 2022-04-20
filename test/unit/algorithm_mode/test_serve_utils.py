# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import io
import json
import os

from mock import MagicMock
import numpy as np
import pytest
from sagemaker_containers.record_pb2 import Record
from sagemaker_containers._recordio import _read_recordio

from sagemaker_algorithm_toolkit import exceptions as exc
from sagemaker_xgboost_container import data_utils
from sagemaker_xgboost_container.constants.sm_env_constants import SAGEMAKER_INFERENCE_ENSEMBLE
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


TEST_RAW_PREDICTIONS = np.array([0.6, 0.1])
TEST_KEYS = [serve_utils.PREDICTED_LABEL, serve_utils.PROBABILITIES]
TEST_PREDICTIONS = [
    {serve_utils.PREDICTED_LABEL: 1, serve_utils.PROBABILITIES: [0.4, 0.6]},
    {serve_utils.PREDICTED_LABEL: 0, serve_utils.PROBABILITIES: [0.9, 0.1]}
]

TEST_KEYS_BINARY_LOG = serve_utils.VALID_OBJECTIVES[serve_utils.BINARY_LOG]
TEST_PREDICTIONS_BINARY_LOG = [
    {serve_utils.PREDICTED_LABEL: 1, serve_utils.LABELS: [0, 1], serve_utils.PROBABILITY: 0.6,
     serve_utils.PROBABILITIES: [0.4, 0.6], serve_utils.RAW_SCORE: 0.6, serve_utils.RAW_SCORES: [0.4, 0.6]},
    {serve_utils.PREDICTED_LABEL: 0, serve_utils.LABELS: [0, 1], serve_utils.PROBABILITY: 0.1,
     serve_utils.PROBABILITIES: [0.9, 0.1], serve_utils.RAW_SCORE: 0.1, serve_utils.RAW_SCORES: [0.9, 0.1]}
]

TEST_RAW_PREDICTIONS_REG_LOG = np.array([0.5, -7.0])
TEST_KEYS_REG_LOG = serve_utils.VALID_OBJECTIVES[serve_utils.REG_LOG]
TEST_PREDICTIONS_REG_LOG = [{"predicted_score": 0.5}, {"predicted_score": -7.0}]


def test_is_selectable_inference_response_false():
    assert not serve_utils.is_selectable_inference_output()


@pytest.fixture(scope='session')
def test_is_selectable_inference_response_true(monkeypatch):
    monkeypatch.setenv('SAGEMAKER_INFERENCE_OUTPUT', serve_utils.PREDICTED_LABEL)
    assert serve_utils.is_selectable_inference_output()


@pytest.fixture(scope='session')
def test_get_selected_content_keys(monkeypatch):
    monkeypatch.setenv('SAGEMAKER_INFERENCE_OUTPUT', serve_utils.PREDICTED_LABEL)
    assert serve_utils.get_selected_output_keys() == [serve_utils.PREDICTED_LABEL]


def test_get_selected_content_keys_error():
    with pytest.raises(RuntimeError):
        serve_utils.get_selected_output_keys()


@pytest.mark.parametrize('test_raw_predictions, selected_keys, objective, expected_predictions', [
    (TEST_RAW_PREDICTIONS, TEST_KEYS_BINARY_LOG, serve_utils.BINARY_LOG, TEST_PREDICTIONS_BINARY_LOG),
    (TEST_RAW_PREDICTIONS_REG_LOG, TEST_KEYS_REG_LOG, serve_utils.REG_LOG, TEST_PREDICTIONS_REG_LOG)
])
def test_get_selected_predictions_all_keys(test_raw_predictions, selected_keys, objective, expected_predictions):
    predictions = serve_utils.get_selected_predictions(test_raw_predictions, selected_keys, objective)
    assert predictions == expected_predictions


def test_get_selected_predictions_nan():
    predictions = serve_utils.get_selected_predictions(np.array([0.6, 32]),
                                                       ["predicted_score", "predicted_label", "foo"],
                                                       serve_utils.REG_LOG)
    assert predictions == [{"predicted_score": 0.6, "predicted_label": np.nan, "foo": np.nan},
                           {"predicted_score": 32, "predicted_label": np.nan, "foo": np.nan}]


def test_get_selected_predictions_invalid_objective():
    with pytest.raises(ValueError):
        serve_utils.get_selected_predictions(TEST_RAW_PREDICTIONS, TEST_KEYS, "rank:pairwise")


@pytest.mark.parametrize('objective, expected_labels, num_class', [
    (serve_utils.BINARY_LOG, [0, 1], ''),
    (serve_utils.MULTI_SOFTPROB, list(range(7)), '7'),
])
def test_get_labels(objective, expected_labels, num_class):
    assert serve_utils._get_labels(objective, num_class=num_class) == expected_labels


def test_get_labels_nan():
    assert np.isnan(serve_utils._get_labels(serve_utils.REG_LOG))


@pytest.mark.parametrize('objective, predictions, expected_predicted_label', [
    (serve_utils.BINARY_HINGE, np.int64(0), 0),
    (serve_utils.BINARY_LOG, np.float64(0.6), 1),
    (serve_utils.BINARY_LOGRAW, np.float64(-7.6), 0),
    (serve_utils.MULTI_SOFTPROB, np.array([0.1, 0.5, 0.4]), 1),
])
def test_get_predicted_label(objective, predictions, expected_predicted_label):
    assert serve_utils._get_predicted_label(objective, predictions) == expected_predicted_label


def test_get_predicted_label_nan():
    assert np.isnan(serve_utils._get_predicted_label(serve_utils.REG_LOG, 0))


@pytest.mark.parametrize('objective, predictions, expected_probability', [
    (serve_utils.BINARY_LOG, np.float64(0.6), 0.6),
    (serve_utils.MULTI_SOFTPROB, np.array([0.1, 0.5, 0.4]), 0.5)
])
def test_get_probability(objective, predictions, expected_probability):
    assert serve_utils._get_probability(objective, predictions) == expected_probability


def test_get_probability_nan():
    assert np.isnan(serve_utils._get_probability(serve_utils.BINARY_HINGE, 0))


@pytest.mark.parametrize('objective, predictions, expected_probabilities', [
    (serve_utils.BINARY_LOG, np.float64(0.6), [0.4, 0.6]),
    (serve_utils.MULTI_SOFTPROB, np.array([0.1, 0.5, 0.4]), [0.1, 0.5, 0.4])
])
def test_get_probabilities(objective, predictions, expected_probabilities):
    assert serve_utils._get_probabilities(objective, predictions) == expected_probabilities


def test_get_probabilities_nan():
    assert np.isnan(serve_utils._get_probabilities(serve_utils.BINARY_HINGE, 0))


@pytest.mark.parametrize('objective, predictions, expected_raw_score', [
    (serve_utils.BINARY_LOG, np.float64(0.6), 0.6),
    (serve_utils.MULTI_SOFTPROB, np.array([0.1, 0.5, 0.4]), 0.5),
    (serve_utils.BINARY_LOGRAW, np.float64(-7.6), -7.6)
])
def test_get_raw_score(objective, predictions, expected_raw_score):
    assert serve_utils._get_raw_score(objective, predictions) == expected_raw_score


def test_get_raw_score_nan():
    assert np.isnan(serve_utils._get_probability(serve_utils.REG_LOG, 0))


@pytest.mark.parametrize('objective, predictions, expected_raw_scores', [
    (serve_utils.BINARY_LOG, np.float64(0.6), [0.4, 0.6]),
    (serve_utils.MULTI_SOFTPROB, np.array([0.1, 0.5, 0.4]), [0.1, 0.5, 0.4]),
    (serve_utils.BINARY_HINGE, np.int64(1), [0, 1])
])
def test_get_raw_scores(objective, predictions, expected_raw_scores):
    assert serve_utils._get_raw_scores(objective, predictions) == expected_raw_scores


def test_get_raw_scores_nan():
    assert np.isnan(serve_utils._get_raw_scores(serve_utils.REG_LOG, 0))


def test_encode_selected_predictions_json():
    expected_json = json.dumps({"predictions": TEST_PREDICTIONS})
    assert serve_utils.encode_selected_predictions(TEST_PREDICTIONS, TEST_KEYS, "application/json") == expected_json


def test_encode_selected_predictions_jsonlines():
    expected_jsonlines = b'{"predicted_label": 1, "probabilities": [0.4, 0.6]}\n' \
                         b'{"predicted_label": 0, "probabilities": [0.9, 0.1]}\n'
    assert serve_utils.encode_selected_predictions(TEST_PREDICTIONS, TEST_KEYS,
                                                   "application/jsonlines") == expected_jsonlines


def test_encode_selected_predictions_protobuf():
    expected_predicted_labels = [[1], [0]]
    expected_probabilities = [[0.4, 0.6], [0.9, 0.1]]

    protobuf_response = serve_utils.encode_selected_predictions(TEST_PREDICTIONS, TEST_KEYS,
                                                                "application/x-recordio-protobuf")
    stream = io.BytesIO(protobuf_response)

    for recordio, predicted_label, probabilities in zip(_read_recordio(stream),
                                                        expected_predicted_labels, expected_probabilities):
        record = Record()
        record.ParseFromString(recordio)
        assert record.label["predicted_label"].float32_tensor.values == predicted_label
        assert all(np.isclose(record.label["probabilities"].float32_tensor.values, probabilities))


def test_encode_selected_predictions_csv():
    expected_csv = '1,"[0.4, 0.6]"\n0,"[0.9, 0.1]"'
    assert serve_utils.encode_selected_predictions(TEST_PREDICTIONS, TEST_KEYS, "text/csv") == expected_csv


def test_encode_selected_content_error():
    with pytest.raises(RuntimeError):
        serve_utils.encode_selected_predictions(TEST_PREDICTIONS, TEST_KEYS, "text/libsvm")


def test_is_ensemble_enabled_var_not_set():
    assert serve_utils.is_ensemble_enabled()


def test_is_ensemble_enabled_var_set_to_false(monkeypatch):
    monkeypatch.setenv(SAGEMAKER_INFERENCE_ENSEMBLE, 'false')
    assert not serve_utils.is_ensemble_enabled()


def test_is_ensemble_enabled_var_set_to_true(monkeypatch):
    monkeypatch.setenv(SAGEMAKER_INFERENCE_ENSEMBLE, 'true')
    assert serve_utils.is_ensemble_enabled()


def test_encode_predictions_as_json_empty_list():
    expected_response = json.dumps({"predictions": []})
    assert expected_response == serve_utils.encode_predictions_as_json([])


def test_encode_predictions_as_json_non_empty_list():
    expected_response = json.dumps({"predictions": [{"score": 0.43861907720565796}, {"score": 0.4533972144126892}]})
    assert expected_response == serve_utils.encode_predictions_as_json([0.43861907720565796, 0.4533972144126892])
