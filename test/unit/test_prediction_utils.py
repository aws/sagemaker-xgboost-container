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
import os
from contextlib import ExitStack as does_not_raise
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from sagemaker_algorithm_toolkit import exceptions as exc
from sagemaker_xgboost_container.prediction_utils import (
    PREDICTIONS_OUTPUT_FILE,
    ValidationPredictionRecorder,
)


@pytest.mark.parametrize(
    "config",
    [
        # binary classification happy path
        dict(
            recorder_kwargs=dict(
                y_true=np.array([1, 0, 1, 0]),
                num_cv_round=1,
                classification=True,
            ),
            record_calls=[
                dict(args=[np.array([0, 1, 2, 3]), np.array([1, 0, 1, 1])]),
            ],
            expected_output=pd.DataFrame(
                {
                    0: [1.0, 0, 1.0, 0],
                    1: [1.0, 0, 1.0, 1.0],
                    2: [1.0, 0, 1.0, 1.0],
                }
            ),
        ),
        # binary classification happy path, multiple folds
        dict(
            recorder_kwargs=dict(
                y_true=np.array([1, 0, 1, 0]),
                num_cv_round=1,
                classification=True,
            ),
            record_calls=[
                dict(args=[np.array([0, 1]), np.array([1, 0])]),
                dict(args=[np.array([2, 3]), np.array([1, 1])]),
            ],
            expected_output=pd.DataFrame(
                {
                    0: [1.0, 0, 1.0, 0],
                    1: [1.0, 0, 1.0, 1.0],
                    2: [1.0, 0, 1.0, 1.0],
                }
            ),
        ),
        # binary classification happy path, multiple repeats
        dict(
            recorder_kwargs=dict(
                y_true=np.array([1, 0, 1, 0]),
                num_cv_round=3,
                classification=True,
            ),
            record_calls=[
                dict(args=[np.array([0, 1, 2, 3]), np.array([0.6, 0, 0.6, 0.6])]),
                dict(args=[np.array([0, 1, 2, 3]), np.array([0.7, 0, 0.7, 0.4])]),
                dict(args=[np.array([0, 1, 2, 3]), np.array([0.8, 0, 0.8, 0.2])]),
            ],
            expected_output=pd.DataFrame(
                {
                    0: [1.0, 0, 1.0, 0],
                    1: [0.7, 0, 0.7, 0.4],
                    2: [1.0, 0, 1.0, 0.0],
                }
            ),
        ),
        # binary classification happy path, multiple repeats
        dict(
            recorder_kwargs=dict(
                y_true=np.array([1, 0, 1, 0]),
                num_cv_round=3,
                classification=True,
            ),
            record_calls=[
                dict(args=[np.array([0, 1, 2]), np.array([0.6, 0, 0.6])]),
                dict(args=[np.array([3]), np.array([0.6])]),
                dict(args=[np.array([0]), np.array([0.7])]),
                dict(args=[np.array([1, 2]), np.array([0, 0.7])]),
                dict(args=[np.array([3]), np.array([0.4])]),
                dict(args=[np.array([0, 1, 2, 3]), np.array([0.8, 0, 0.8, 0.2])]),
            ],
            expected_output=pd.DataFrame(
                {
                    0: [1.0, 0, 1.0, 0],
                    1: [0.7, 0, 0.7, 0.4],
                    2: [1.0, 0, 1.0, 0.0],
                }
            ),
        ),
        # regression happy path, multiple repeats
        dict(
            recorder_kwargs=dict(
                y_true=np.array([0.8, 0, 1, 0]),
                num_cv_round=3,
                classification=False,
            ),
            record_calls=[
                dict(args=[np.array([0, 1, 2, 3]), np.array([0.6, 0, 0.6, 0.6])]),
                dict(args=[np.array([0, 1, 2, 3]), np.array([0.7, 0, 0.7, 0.4])]),
                dict(args=[np.array([0, 1, 2, 3]), np.array([0.8, 0, 0.8, 0.2])]),
            ],
            expected_output=pd.DataFrame(
                {
                    0: [0.8, 0, 1.0, 0],
                    1: [0.7, 0, 0.7, 0.4],
                }
            ),
        ),
        # multiclass classification happy path
        dict(
            recorder_kwargs=dict(
                y_true=np.array([1, 2, 1, 0]),
                num_cv_round=1,
                classification=True,
            ),
            record_calls=[
                dict(
                    args=[
                        np.array([0, 1, 2, 3]),
                        np.array([[0.1, 0.6, 0.3, 0.1], [0.1, 0.3, 0.4, 0.1], [0.8, 0.1, 0.3, 0.8]]).T,
                    ]
                ),
            ],
            expected_output=pd.DataFrame(
                {
                    0: [1.0, 2.0, 1.0, 0.0],
                    1: [0.8, 0.6, 0.4, 0.8],
                    2: [2.0, 0.0, 1.0, 2.0],
                }
            ),
        ),
        # incorrect shape of predictions
        dict(
            recorder_kwargs=dict(
                y_true=np.array([0.8, 0, 1, 0]),
                num_cv_round=3,
                classification=False,
            ),
            record_calls=[
                dict(args=[np.array([0, 1, 2, 3]), np.array([0.6, 0, 0.6, 0.6])]),
                dict(
                    args=[
                        np.array([0, 1, 2, 3]),
                        np.array([[0.1, 0.6, 0.3, 0.1], [0.1, 0.3, 0.4, 0.1], [0.8, 0.1, 0.3, 0.8]]).T,
                    ],
                    record_raises=pytest.raises(exc.AlgorithmError),
                ),
            ],
            expected_output=None,
        ),
        # incomplete predictions
        dict(
            recorder_kwargs=dict(
                y_true=np.array([0.8, 0, 1, 0]),
                num_cv_round=1,
                classification=False,
            ),
            record_calls=[
                dict(args=[np.array([0, 1, 2]), np.array([0.6, 0, 0.3])]),
            ],
            save_raises=pytest.raises(exc.AlgorithmError),
            expected_output=None,
        ),
        # too many predictions
        dict(
            recorder_kwargs=dict(
                y_true=np.array([0.8, 0, 1, 0]),
                num_cv_round=1,
                classification=False,
            ),
            record_calls=[
                dict(args=[np.array([0, 1, 2]), np.array([0.6, 0, 0.3])]),
                dict(
                    args=[np.array([0, 1, 2]), np.array([0.6, 0, 0.3])], record_raises=pytest.raises(exc.AlgorithmError)
                ),
            ],
            expected_output=None,
        ),
    ],
)
def test_validation_prediction_recorder(config):
    with TemporaryDirectory() as temp_folder:
        recorder = ValidationPredictionRecorder(output_data_dir=temp_folder, **config["recorder_kwargs"])
        for call_config in config["record_calls"]:
            with call_config.get("record_raises", does_not_raise()):
                recorder.record(*call_config["args"])
            if call_config.get("record_raises", None) is not None:
                return

        with config.get("save_raises", does_not_raise()):
            recorder.save()
        if config.get("save_raises", None) is not None:
            return
        df = pd.read_csv(os.path.join(temp_folder, PREDICTIONS_OUTPUT_FILE), header=None)
        assert df.equals(config["expected_output"])
