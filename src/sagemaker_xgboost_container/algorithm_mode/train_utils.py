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

from sagemaker_xgboost_container.metrics.custom_metrics import (
    configure_feval,
    get_custom_metrics,
)

HPO_SEPARATOR = ":"


# These are helper functions for parsing the list of metrics to be outputted
def get_union_metrics(metric_a, metric_b):
    """Union of metric_a and metric_b

    :param metric_a: list
    :param metric_b: list
    :return: Union metrics list from metric_a and metric_b
    """
    if metric_a is None and metric_b is None:
        return None
    elif metric_a is None:
        return metric_b
    elif metric_b is None:
        return metric_a
    else:
        # The order of metric_list need to be consistent among all hosts in distributed training
        # So we have metric_list sorted here.
        metric_list = sorted(list(set(metric_a).union(metric_b)))
        return metric_list


def get_eval_metrics_and_feval(tuning_objective_metric_param, eval_metric):
    """Return list of default xgb evaluation metrics and list of container defined metrics.

    XGB uses the 'eval_metric' parameter for the evaluation metrics supported by default, and 'feval' as an argument
    during training to validate using custom evaluation metrics. The argument 'feval' takes a function as value; the
    method returned here will be configured to run for only the metrics the user specifies.

    :param tuning_objective_metric_param: HPO metric
    :param eval_metric: list of xgb metrics to output
    :return: cleaned list of xgb supported evaluation metrics, method configured with container defined metrics,
    and tuning objective metric.
    """
    tuning_objective_metric = None
    configured_eval = None
    cleaned_eval_metrics = None

    if tuning_objective_metric_param is not None:
        tuning_objective_metric_tuple = MetricNameComponents.decode(tuning_objective_metric_param)
        tuning_objective_metric = tuning_objective_metric_tuple.metric_name.split(",")
        logging.info("Setting up HPO optimized metric to be : {}".format(tuning_objective_metric_tuple.metric_name))

    union_metrics = get_union_metrics(tuning_objective_metric, eval_metric)

    if union_metrics is not None:
        feval_metrics = get_custom_metrics(union_metrics)
        if feval_metrics:
            configured_eval = configure_feval(feval_metrics)
            cleaned_eval_metrics = list(set(union_metrics) - set(feval_metrics))
        else:
            cleaned_eval_metrics = union_metrics

    return cleaned_eval_metrics, configured_eval, tuning_objective_metric


def cleanup_dir(dir, file_prefix):
    """Clean up directory

    This function is used to remove extra files from a directory other than 'file'.

    :param dir: model directory which needs to be cleaned
    :param file_prefix: file name prefix which isn't removed if present
    """

    def _format_path(file_name):
        return os.path.join(dir, file_name)

    def _remove(path):
        try:
            os.remove(path)
        except Exception:
            pass

    for data_file in os.listdir(dir):
        path = _format_path(data_file)
        if os.path.isfile(path) and not data_file.startswith(file_prefix):
            _remove(path)


class MetricNameComponents(object):
    def __init__(self, data_segment, metric_name, emission_frequency=None):
        self.data_segment = data_segment
        self.metric_name = metric_name
        self.emission_frequency = emission_frequency

    @classmethod
    def decode(cls, tuning_objective_metric):
        result = tuning_objective_metric.split(":")
        return MetricNameComponents(*result)


def _get_bytes_to_mb(num_bytes):
    return round(num_bytes / (1024 * 1024), 2)
