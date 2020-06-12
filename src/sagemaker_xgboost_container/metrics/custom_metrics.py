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
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error


# TODO: Rename both according to AutoML standards
def accuracy(preds, dtrain):
    """Compute accuracy.

    :param preds: Prediction values
    :param dtrain: Training data with labels
    :return: Metric name, accuracy value.
    """
    labels = dtrain.get_label()
    rounded_preds = [np.argmax(value) if (type(value) is np.ndarray) else round(value) for value in preds]
    return 'accuracy', accuracy_score(labels, rounded_preds)


def f1(preds, dtrain):
    """Compute f1 score. This can be used for multiclassification training.

    For more information see: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

    :param preds: Prediction values
    :param dtrain: Training data with labels
    :return: Metric name, f1 score
    """
    labels = dtrain.get_label()
    rounded_preds = [np.argmax(value) if (type(value) is np.ndarray) else round(value) for value in preds]
    return 'f1', f1_score(labels, rounded_preds, average='macro')


def mse(preds, dtrain):
    """Compute mean squared error.

    For more information see: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    :param preds: Prediction values
    :param dtrain: Training data with labels
    :return: Metric name, mean squared error
    """
    labels = dtrain.get_label()
    return 'mse', mean_squared_error(labels, preds)


CUSTOM_METRICS = {
    "accuracy": accuracy,
    "f1": f1,
    "mse": mse
}


def get_custom_metrics(eval_metrics):
    """Get container defined metrics from metrics list."""
    return set(eval_metrics).intersection(CUSTOM_METRICS.keys())


def configure_feval(custom_metric_list):
    """Configure custom_feval method with metrics specified by user.

    XGBoost.train() can take a feval argument whose value is a function. This method configures that function with
    multipl metrics if required, then returns to use during training.

    :param custom_metric_list: Metrics to evaluate using feval
    :return: Configured feval method
    """
    def custom_feval(preds, dtrain):
        metrics = []

        for metric_method_name in custom_metric_list:
            custom_metric = CUSTOM_METRICS[metric_method_name]
            metrics.append(custom_metric(preds, dtrain))

        return metrics

    return custom_feval
