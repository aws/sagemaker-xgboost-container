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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


# From 1.2, custom evaluation metric receives raw prediction.
# For binary classification (binary:logistic as objective),
# the raw prediction is log-odds, which can be translated to
# probability by sigmoid function.
# https://github.com/dmlc/xgboost/releases/tag/v1.2.0
def sigmoid(x):
    """Transform binary classification margin output to probability
    Instead of exp(-x), we employ tanh as it is stable, fast, and fairly accurate."""
    return 0.5 * (1 + np.tanh(0.5 * x))


def margin_to_class_label(preds):
    """Converts raw margin output to class label. Instead of converting margin output to
    probability as intermediate step, we compare in log-odds space (i.e. check if log-odds > 0)."""
    if type(preds[0]) is np.ndarray:
        return np.argmax(preds, axis=-1)
    else:
        return (preds > 0.0).astype(int)


# TODO: Rename both according to AutoML standards
def accuracy(preds, dtrain):
    """Compute accuracy.

    :param preds: Prediction values
    :param dtrain: Training data with labels
    :return: Metric name, accuracy value.
    """
    return "accuracy", compute_multiclass_and_binary_metrics(accuracy_score, preds, dtrain)


def balanced_accuracy(preds, dtrain):
    """Compute balanced accuracy.

    :param preds: Prediction values
    :param dtrain: Training data with labels
    :return: Metric name, balanced accuracy value.
    """
    return "balanced_accuracy", compute_multiclass_and_binary_metrics(balanced_accuracy_score, preds, dtrain)


def f1(preds, dtrain):
    """Compute f1 score. This can be used for multiclassification training.
    For more information see: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    :param preds: Prediction values
    :param dtrain: Training data with labels
    :return: Metric name, f1 score
    """
    return "f1", compute_multiclass_and_binary_metrics(lambda x, y: f1_score(x, y, average="macro"), preds, dtrain)


def f1_binary(preds, dtrain):
    """Compute f1 binary score. This can be used for binaryclassification training.

    For more information see: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

    :param preds: Prediction values
    :param dtrain: Training data with labels
    :return: Metric name, f1 score
    """
    return "f1_binary", compute_multiclass_and_binary_metrics(
        lambda x, y: f1_score(x, y, average="binary"), preds, dtrain
    )


def f1_macro(preds, dtrain):
    """Compute f1 macro score. This can be used for multiclassification training.

    For more information see: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

    :param preds: Prediction values
    :param dtrain: Training data with labels
    :return: Metric name, f1 score
    """
    return "f1_macro", compute_multiclass_and_binary_metrics(
        lambda x, y: f1_score(x, y, average="macro"), preds, dtrain
    )


def mae(preds, dtrain):
    """Compute mean absolute error.
    For more information see: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
    :param preds: Prediction values
    :param dtrain: Training data with labels
    :return: Metric name, mean absolute error
    """
    labels = dtrain.get_label()
    return "mae", mean_absolute_error(labels, preds)


def mse(preds, dtrain):
    """Compute mean squared error.

    For more information see: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    :param preds: Prediction values
    :param dtrain: Training data with labels
    :return: Metric name, mean squared error
    """
    labels = dtrain.get_label()
    return "mse", mean_squared_error(labels, preds)


def rmse(preds, dtrain):
    """Compute mean squared error.
    For more information see: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    :param preds: Prediction values
    :param dtrain: Training data with labels
    :return: Metric name, root mean squared error
    """
    labels = dtrain.get_label()
    return "rmse", mean_squared_error(labels, preds, squared=False)


def precision(preds, dtrain):
    """Compute precision.

    :param preds: Prediction values
    :param dtrain: Training data with labels
    :return: Metric name, precision value.
    """
    return "precision", compute_multiclass_and_binary_metrics(precision_score, preds, dtrain)


def precision_macro(preds, dtrain):
    """Compute precision macro score. This can be used for multiclassification training.

    :param preds: Prediction values
    :param dtrain: Training data with labels
    :return: Metric name, precision score
    """
    return "precision_macro", compute_multiclass_and_binary_metrics(
        lambda x, y: precision_score(x, y, average="macro"), preds, dtrain
    )


def precision_micro(preds, dtrain):
    """Compute precision micro score. This can be used for multiclassification training.

    :param preds: Prediction values
    :param dtrain: Training data with labels
    :return: Metric name, precision score
    """
    return "precision_micro", compute_multiclass_and_binary_metrics(
        lambda x, y: precision_score(x, y, average="micro"), preds, dtrain
    )


def recall(preds, dtrain):
    """Compute recall.

    :param preds: Prediction values
    :param dtrain: Training data with labels
    :return: Metric name, recall value.
    """
    return "recall", compute_multiclass_and_binary_metrics(recall_score, preds, dtrain)


def recall_macro(preds, dtrain):
    """Compute recall macro score. This can be used for multiclassification training.

    :param preds: Prediction values
    :param dtrain: Training data with labels
    :return: Metric name, recall score
    """
    return "recall_macro", compute_multiclass_and_binary_metrics(
        lambda x, y: recall_score(x, y, average="macro"), preds, dtrain
    )


def recall_micro(preds, dtrain):
    """Compute recall micro score. This can be used for multiclassification training.

    :param preds: Prediction values
    :param dtrain: Training data with labels
    :return: Metric name, recall score
    """
    return "recall_micro", compute_multiclass_and_binary_metrics(
        lambda x, y: recall_score(x, y, average="micro"), preds, dtrain
    )


def r2(preds, dtrain):
    """Compute R^2 (coefficient of determination) regression score.
    For more information see: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
    :param preds: Prediction values
    :param dtrain: Training data with labels
    :return: Metric name, coefficient of determination
    """
    labels = dtrain.get_label()
    return "r2", r2_score(labels, preds)


def compute_multiclass_and_binary_metrics(metricfunc, preds, dtrain):
    """Compute multiclass and binary metrics based on metric calculator function defined in metricfunc
    :param preds: Prediction values
    :param dtrain: Training data with labels
    :return: Metric score calculated by 'metricfunc'
    """
    score = 0.0
    if preds.size > 0:
        labels = dtrain.get_label()
        pred_labels = margin_to_class_label(preds)
        score = metricfunc(labels, pred_labels)
    return score


CUSTOM_METRICS = {
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "f1": f1,
    "f1_binary": f1_binary,
    "f1_macro": f1_macro,
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
    "precision": precision,
    "precision_macro": precision_macro,
    "precision_micro": precision_micro,
    "r2": r2,
    "recall": recall,
    "recall_macro": recall_macro,
    "recall_micro": recall_micro,
}


def get_custom_metrics(eval_metrics):
    """Get container defined metrics from metrics list.

    The order of the returning custom metrics need to be consistent with the input for distributed training.
    Otherwise, metrics reported from each host will be miscalculated in the master host. (P70679777)
    """
    return [eval_m for eval_m in eval_metrics if eval_m in CUSTOM_METRICS.keys()]


def configure_feval(custom_metric_list):
    """Configure custom_feval method with metrics specified by user.

    XGBoost.train() can take a feval argument whose value is a function. This method configures that function with
    multiple metrics if required, then returns to use during training.

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
