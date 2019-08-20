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
import re
from typing import Tuple, Optional
import xgboost
from sagemaker_xgboost_container.metrics.custom_metrics import get_custom_metrics, configure_feval
from sagemaker_xgboost_container.algorithm_mode.callback import CHECKPOINT_FILENAME, CHECKPOINT_NUM_DIGITS


HPO_SEPARATOR = ':'


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
        metric_list = list(set(metric_a).union(metric_b))
        return metric_list


def get_eval_metrics_and_feval(tuning_objective_metric_param, eval_metric):
    """Return list of default xgb evaluation metrics and list of container defined metrics.

    XGB uses the 'eval_metric' parameter for the evaluation metrics supported by default, and 'feval' as an argument
    during training to validate using custom evaluation metrics. The argument 'feval' takes a function as value; the
    method returned here will be configured to run for only the metrics the user specifies.

    :param tuning_objective_metric_param: HPO metric
    :param eval_metric: list of xgb metrics to output
    :return: cleaned list of xgb supported evaluation metrics, method configured with container defined metrics.
    """
    tuning_objective_metric = None
    configured_eval = None
    cleaned_eval_metrics = None

    if tuning_objective_metric_param is not None:
        tuning_objective_metric_tuple = MetricNameComponents.decode(tuning_objective_metric_param)
        tuning_objective_metric = tuning_objective_metric_tuple.metric_name.split(',')
        logging.info('Setting up HPO optimized metric to be : {}'.format(tuning_objective_metric_tuple.metric_name))

    union_metrics = get_union_metrics(tuning_objective_metric, eval_metric)

    if union_metrics is not None:
        feval_metrics = get_custom_metrics(union_metrics)
        if feval_metrics:
            configured_eval = configure_feval(feval_metrics)
            cleaned_eval_metrics = list(set(union_metrics) - set(feval_metrics))
        else:
            cleaned_eval_metrics = union_metrics

    return cleaned_eval_metrics, configured_eval


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


def load_checkpoint(checkpoint_dir: str, max_try=5) -> Tuple[Optional[str], int]:
    """
    :param checkpoint_dir: e.g., /opt/ml/checkpoints
    :return xgb_model: file path of stored xgb model. None if no checkpoint.
    :return iteration: iterations completed before last checkpoiint.
    """
    if not os.path.exists(checkpoint_dir):
        return None, 0

    regex = r"^{0}\.\d{{{1}}}$".format(CHECKPOINT_FILENAME, CHECKPOINT_NUM_DIGITS)
    checkpoints = [f for f in os.listdir(checkpoint_dir) if re.match(regex, f)]
    if not checkpoints:
        return None, 0
    checkpoints.sort()

    xgb_model, iteration = None, 0

    for _ in range(max_try):
        try:
            latest_checkpoint = checkpoints.pop()
            xgb_model = os.path.join(checkpoint_dir, latest_checkpoint)
            booster = xgboost.Booster()
            booster.load_model(xgb_model)

            filename, extension = latest_checkpoint.split('.')
            iteration = int(extension) + 1
            break
        except xgboost.core.XGBoostError:
            logging.debug("Wrong checkpoint model format %s", latest_checkpoint)

    return xgb_model, iteration
