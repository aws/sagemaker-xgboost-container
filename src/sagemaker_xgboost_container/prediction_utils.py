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
import logging
import os

import numpy as np
from scipy import stats

from sagemaker_algorithm_toolkit import exceptions as exc

PREDICTIONS_OUTPUT_FILE = "predictions.csv"
EXAMPLE_ROWS_EXCEPTION_COUNT = 100


class ValidationPredictionRecorder:
    """Helper class to record and store predictions obtained on different train / validation
    folds. Predictions are stored in folder specified by SM_OUTPUT_DATA_DIR env variable set by
    training platform, and sometimes modified by container code. Additional artefacts at the
    end of the training job are stored in output s3 path as output.tar.gz.

    Attributes:
        y_true           (1d numpy array): Ground truth labels.
        num_cv_round     (int): number times cross validation procedure will be repeated.
        classification   (bool): indicates type of learning problem.
    """

    def __init__(self, y_true: np.ndarray, num_cv_round: int, classification: bool, output_data_dir: str) -> None:
        self.y_true = y_true.copy()
        num_rows = len(y_true)
        self.num_cv_round = num_cv_round
        self.y_pred = np.zeros((num_rows, num_cv_round))
        self.y_prob = self.y_pred.copy() if classification else None
        self.cv_repeat_counter = np.zeros((num_rows,)).astype(int)
        self.classification = classification
        self.output_data_dir = output_data_dir
        self.pred_ndim_ = None

    def record(self, indices: np.ndarray, predictions: np.ndarray) -> None:
        """Record predictions on a single validation fold in-memory.

        :param indices: indicates for which rows the predictions were made.
        :param predictions: predictions for rows specified in `indices` variable.
        """
        if self.pred_ndim_ is None:
            self.pred_ndim_ = predictions.ndim
        if self.pred_ndim_ != predictions.ndim:
            raise exc.AlgorithmError(f"Expected predictions with ndim={self.pred_ndim_}, got ndim={predictions.ndim}.")

        cv_repeat_idx = self.cv_repeat_counter[indices]
        if np.any(cv_repeat_idx == self.num_cv_round):
            sample_rows = cv_repeat_idx[cv_repeat_idx == self.num_cv_round]
            sample_rows = sample_rows[:EXAMPLE_ROWS_EXCEPTION_COUNT]
            raise exc.AlgorithmError(
                f"More than {self.num_cv_round} repeated predictions for same row were provided. "
                f"Example row indices where this is the case: {sample_rows}."
            )

        if self.classification:
            if predictions.ndim > 1:
                labels = np.argmax(predictions, axis=-1)
                proba = predictions[np.arange(len(labels)), labels]
            else:
                labels = 1 * (predictions > 0.5)
                proba = predictions
            self.y_pred[indices, cv_repeat_idx] = labels
            self.y_prob[indices, cv_repeat_idx] = proba
        else:
            self.y_pred[indices, cv_repeat_idx] = predictions
        self.cv_repeat_counter[indices] += 1

    def _aggregate_predictions(self) -> np.ndarray:
        if not np.all(self.cv_repeat_counter == self.num_cv_round):
            sample_rows = self.cv_repeat_counter[self.cv_repeat_counter != self.num_cv_round]
            sample_rows = sample_rows[:EXAMPLE_ROWS_EXCEPTION_COUNT]
            raise exc.AlgorithmError(
                f"For some rows number of repeated validation set predictions provided is not {self.num_cv_round}. "
                f"Example row indices where this is the case: {sample_rows}"
            )

        columns = [self.y_true]
        if self.classification:
            columns.append(self.y_prob.mean(axis=-1))
            # mode always returns same number of dimensions of output as for input
            model_result = stats.mode(self.y_pred, axis=1, keepdims=True)
            model_values = model_result.mode
            if model_values.ndim > 1:
                model_values = model_values[:, 0]
            columns.append(model_values)
        else:
            columns.append(self.y_pred.mean(axis=-1))

        return np.vstack(columns).T

    def _check_output_path(self) -> None:
        if not os.path.exists(self.output_data_dir):
            logging.warn(f"Output directory {self.output_data_dir} not found; Creating the output directory.")
            os.makedirs(self.output_data_dir)

    def _get_save_path(self) -> str:
        return os.path.join(self.output_data_dir, PREDICTIONS_OUTPUT_FILE)

    def save(self) -> None:
        """Serialize predictions as .csv file in output data directory."""
        self._check_output_path()
        save_path = self._get_save_path()

        logging.info(f"Storing predictions on validation set(s) in {save_path}")
        np.savetxt(save_path, self._aggregate_predictions(), delimiter=",", fmt="%f")
