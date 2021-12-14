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
import logging
import numpy as np
from scipy import stats

PREDICTIONS_OUTPUT_FILE = 'predictions.csv'


class ValidationPredictionRecorder:
    """Helper class to record and store predictions obtained on different train / validation
    folds. Predictions are stored in additional artefact folder, and as a result are stored
    in output path of training job as output.tar.gz.

    Attributes:
        y_true           (1d numpy array): Ground truth labels.
        num_cv_round     (int): number times cross validation procedure will be repeated.
        classification   (bool): indicates type of learning problem.
    """
    def __init__(self, y_true: np.ndarray, num_cv_round: int, classification: bool, output_data_dir: str):
        self.y_true = y_true.copy()
        num_rows = len(y_true)
        self.y_pred = np.zeros((num_rows, num_cv_round))
        self.y_prob = self.y_pred.copy() if classification else None
        self.cv_repeat_counter = np.zeros((num_rows,)).astype(int)
        self.classification = classification
        self.output_data_dir = output_data_dir

    def record(self, indices: np.ndarray, predictions: np.ndarray):
        """Record predictions on a single validation fold in-memory.

        If current host is master host, initialize and start the Rabit Tracker in the background. All hosts then connect
        to the master host to set up Rabit rank.

        :param indices: indicates for which rows the predictions were made.
        :param predictions: predictions for rows specified in `indices` variable.
        """
        cv_repeat_idx = self.cv_repeat_counter[indices]
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

    def save(self):
        """Serialize predictions to this instance's output data directory."""
        columns = [self.y_true]
        if self.classification:
            columns.append(self.y_prob.mean(axis=-1))
            # mode always returns same number of dimensions of output as for input
            columns.append(stats.mode(self.y_pred, axis=1).mode[:, 0])
        else:
            columns.append(self.y_pred.mean(axis=-1))

        if not os.path.exists(self.output_data_dir):
            os.makedirs(self.output_data_dir)

        pred_path = os.path.join(self.output_data_dir, PREDICTIONS_OUTPUT_FILE)
        logging.info(f"Storing validation set(s) predictions in {pred_path}")
        np.savetxt(pred_path, np.hstack(columns), delimiter=',', fmt='%f')
