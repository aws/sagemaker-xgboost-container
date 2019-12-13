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
from sagemaker_algorithm_toolkit import metrics as m

from sagemaker_xgboost_container.constants.xgb_constants import XGB_MAXIMIZE_METRICS, XGB_MINIMIZE_METRICS


# https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-tuning.html
def initialize():

    maximize_metrics = [
        m.Metric(name="validation:{}".format(metric_name),
                 direction=m.Metric.MAXIMIZE,
                 regex=".*\\[[0-9]+\\].*#011validation-{}:(\\S+)".format(metric_name))
        for metric_name in XGB_MAXIMIZE_METRICS]

    minimize_metrics = [
        m.Metric(name="validation:{}".format(metric_name),
                 direction=m.Metric.MINIMIZE,
                 regex=".*\\[[0-9]+\\].*#011validation-{}:(\\S+)".format(metric_name))
        for metric_name in XGB_MINIMIZE_METRICS]

    metrics = maximize_metrics + minimize_metrics
    return m.Metrics(*metrics)
