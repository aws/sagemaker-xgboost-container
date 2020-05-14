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
import io
import json
import numpy as np
import os
import unittest

from sagemaker_containers.record_pb2 import Record
from sagemaker_containers._recordio import _read_recordio

from sagemaker_xgboost_container.algorithm_mode import serve_utils

TEST_DATA = [0.6, 0.1]
TEST_KEYS = [serve_utils.PREDICTED_LABEL, serve_utils.PROBABILITIES]
TEST_CONTENT = [
    {serve_utils.PREDICTED_LABEL: 1, serve_utils.PROBABILITIES: [0.4, 0.6]},
    {serve_utils.PREDICTED_LABEL: 0, serve_utils.PROBABILITIES: [0.9, 0.1]}
]
TEST_JSON = json.dumps({'predictions': TEST_CONTENT})

TEST_KEYS_BINARY_LOG = serve_utils.VALID_OBJECTIVES[serve_utils.BINARY_LOG]
TEST_CONTENT_BINARY_LOG = [
    {serve_utils.PREDICTED_LABEL: 1, serve_utils.LABELS: [0, 1], serve_utils.PROBABILITY: 0.6,
     serve_utils.PROBABILITIES: [0.4, 0.6], serve_utils.RAW_SCORE: 0.6, serve_utils.RAW_SCORES: [0.4, 0.6]},
    {serve_utils.PREDICTED_LABEL: 0, serve_utils.LABELS: [0, 1], serve_utils.PROBABILITY: 0.1,
     serve_utils.PROBABILITIES: [0.9, 0.1], serve_utils.RAW_SCORE: 0.1, serve_utils.RAW_SCORES: [0.9, 0.1]}
]

TEST_DATA_REG_LOG = [0.5, -7.0]
TEST_KEYS_REG_LOG = serve_utils.VALID_OBJECTIVES[serve_utils.REG_LOG]
TEST_CONTENT_REG_LOG = [{"predicted_score": 0.5}, {"predicted_score": -7.0}]


class TestServeUtils(unittest.TestCase):

    def test_is_selectable_inference_response_false(self):
        assert not serve_utils.is_selectable_inference_response()

    def test_is_selectable_inference_response_true(self):
        os.environ['SAGEMAKER_INFERENCE_OUTPUT'] = 'predicted_label'
        assert serve_utils.is_selectable_inference_response()
        del os.environ['SAGEMAKER_INFERENCE_OUTPUT']

    def test_get_selected_content_keys(self):
        os.environ['SAGEMAKER_INFERENCE_OUTPUT'] = 'predicted_label'
        assert serve_utils.get_selected_content_keys() == ['predicted_label']
        del os.environ['SAGEMAKER_INFERENCE_OUTPUT']

    def test_get_selected_content_keys_error(self):
        with self.assertRaises(RuntimeError):
            serve_utils.get_selected_content_keys()

    def test_get_selected_content_classification_all_keys(self):
        content = serve_utils.get_selected_content(TEST_DATA, TEST_KEYS_BINARY_LOG, serve_utils.BINARY_LOG)
        assert content == TEST_CONTENT_BINARY_LOG

    def test_get_selected_content_regression_all_keys(self):
        content = serve_utils.get_selected_content(TEST_DATA_REG_LOG, TEST_KEYS_REG_LOG, serve_utils.REG_LOG)
        assert content == TEST_CONTENT_REG_LOG

    def test_get_selected_content_nan(self):
        pass

    def test_get_selected_content_invalid_objective(self):
        with self.assertRaises(ValueError):
            serve_utils.get_selected_content(TEST_DATA, TEST_KEYS, "rank:pairwise")

    def test_get_labels_binary(self):
        assert serve_utils._get_labels(serve_utils.BINARY_LOG) == [0, 1]

    def test_get_labels_multi(self):
        assert serve_utils._get_labels(serve_utils.MULTI_SOFTPROB, num_class='7') == list(range(7))

    def test_get_labels_nan(self):
        assert np.isnan(serve_utils._get_labels(serve_utils.REG_LOG))

    def test_get_predicted_label_hinge(self):
        assert serve_utils._get_predicted_label(serve_utils.BINARY_HINGE, 0) == 0

    def test_get_predicted_label_log(self):
        assert serve_utils._get_predicted_label(serve_utils.BINARY_LOG, 0.6) == 1

    def test_get_predicted_label_raw(self):
        assert serve_utils._get_predicted_label(serve_utils.BINARY_LOGRAW, -7.6) == 0

    def test_get_predicted_label_softprob(self):
        assert serve_utils._get_predicted_label(serve_utils.MULTI_SOFTPROB, [0.1, 0.5, 0.4]) == 1

    def test_get_predicted_label_nan(self):
        assert np.isnan(serve_utils._get_predicted_label(serve_utils.REG_LOG, 0))

    def test_get_probability(self):
        assert serve_utils._get_probability(serve_utils.BINARY_LOG, 0.6) == 0.6

    def test_get_probability_softprob(self):
        assert serve_utils._get_probability(serve_utils.MULTI_SOFTPROB, [0.1, 0.5, 0.4]) == 0.5

    def test_get_probability_nan(self):
        assert np.isnan(serve_utils._get_probability(serve_utils.REG_LOG, 0))

    def test_get_probabilities(self):
        assert serve_utils._get_probabilities(serve_utils.BINARY_LOG, 0.6) == [0.4, 0.6]

    def test_get_probabilities_softprob(self):
        assert serve_utils._get_probabilities(serve_utils.MULTI_SOFTPROB, [0.1, 0.5, 0.4]) == [0.1, 0.5, 0.4]

    def test_get_probabilities_nan(self):
        assert np.isnan(serve_utils._get_probabilities(serve_utils.REG_LOG, 0))

    def test_get_raw_score(self):
        assert serve_utils._get_raw_score(serve_utils.BINARY_LOGRAW, -7.6) == -7.6

    def test_get_raw_score_softprob(self):
        assert serve_utils._get_probability(serve_utils.MULTI_SOFTPROB, [0.1, 0.5, 0.4]) == 0.5

    def test_get_raw_score_nan(self):
        assert np.isnan(serve_utils._get_probability(serve_utils.REG_LOG, 0))

    def test_get_raw_scores(self):
        assert serve_utils._get_raw_scores(serve_utils.BINARY_LOGRAW, -7.6) == [8.6, -7.6]

    def test_get_raw_scores_softprob(self):
        assert serve_utils._get_raw_scores(serve_utils.MULTI_SOFTPROB, [0.1, 0.5, 0.4]) == [0.1, 0.5, 0.4]

    def test_get_raw_scores_nan(self):
        assert np.isnan(serve_utils._get_probabilities(serve_utils.REG_LOG, 0))

    def test_encode_selected_content_json(self):
        expected_json = json.dumps({"predictions": TEST_CONTENT})
        assert serve_utils.encode_selected_content(TEST_CONTENT, TEST_KEYS, "application/json") == expected_json

    def test_encode_selected_content_jsonlines(self):
        expected_jsonlines = b'{\"predicted_label\": 1, \"probabilities\": [0.4, 0.6]}\n' \
                             b'{\"predicted_label\": 0, \"probabilities": [0.9, 0.1]}\n'
        assert serve_utils.encode_selected_content(TEST_CONTENT, TEST_KEYS,
                                                   "application/jsonlines") == expected_jsonlines

    def test_encode_selected_content_protobuf(self):
        expected_predicted_labels = [[1], [0]]
        expected_probabilities = [[0.4, 0.6], [0.9, 0.1]]

        protobuf_response = serve_utils.encode_selected_content(TEST_CONTENT, TEST_KEYS,
                                                                "application/x-recordio-protobuf")
        stream = io.BytesIO(protobuf_response)

        for recordio, predicted_label, probabilities in zip(_read_recordio(stream),
                                                            expected_predicted_labels, expected_probabilities):
            record = Record()
            record.ParseFromString(recordio)
            assert record.label["predicted_label"].float32_tensor.values == predicted_label
            assert all(np.isclose(record.label["probabilities"].float32_tensor.values, probabilities))

    def test_encode_selected_content_csv(self):
        expected_csv = '1,"[0.4, 0.6]"\r\n0,"[0.9, 0.1]"\r\n'
        assert serve_utils.encode_selected_content(TEST_CONTENT, TEST_KEYS, "text/csv") == expected_csv

    def test_encode_selected_content_error(self):
        with self.assertRaises(RuntimeError):
            serve_utils.encode_selected_content(TEST_CONTENT, TEST_KEYS, "text/libsvm")

    def test_encoder_jsonlines_from_json(self):
        json_response = json.dumps({'predictions': TEST_CONTENT})
        expected_jsonlines = b'{\"predicted_label\": 1, \"probabilities\": [0.4, 0.6]}\n' \
                             b'{\"predicted_label\": 0, \"probabilities": [0.9, 0.1]}\n'

        jsonlines_response = serve_utils.encoder_jsonlines_from_json(json_response)
        self.assertEqual(expected_jsonlines, jsonlines_response)

    def test_encoder_jsonlines_from_json_error(self):
        bad_json_response = json.dumps({'predictions': [], 'metadata': []})
        with self.assertRaises(ValueError):
            serve_utils.encoder_jsonlines_from_json(bad_json_response)
