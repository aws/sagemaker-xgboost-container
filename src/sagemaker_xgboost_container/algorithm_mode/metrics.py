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


# https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-tuning.html
def initialize():
    return m.Metrics(
        m.Metric(name="validation:accuracy",
                 direction=m.Metric.MAXIMIZE,
                 regex=".*\\[[0-9]+\\].*#011validation-accuracy:(\\S+)"),
        m.Metric(name="validation:auc",
                 direction=m.Metric.MAXIMIZE,
                 regex=".*\\[[0-9]+\\].*#011validation-auc:(\\S+)"),
        m.Metric(name="validation:error",
                 direction=m.Metric.MINIMIZE,
                 regex=".*\\[[0-9]+\\].*#011validation-error:(\\S+)"),
        m.Metric(name="validation:f1",
                 direction=m.Metric.MAXIMIZE,
                 regex=".*\\[[0-9]+\\].*#011validation-f1:(\\S+)"),
        m.Metric(name="validation:logloss",
                 direction=m.Metric.MINIMIZE,
                 regex=".*\\[[0-9]+\\].*#011validation-logloss:(\\S+)"),
        m.Metric(name="validation:mae",
                 direction=m.Metric.MINIMIZE,
                 regex=".*\\[[0-9]+\\].*#011validation-mae:(\\S+)"),
        m.Metric(name="validation:map",
                 direction=m.Metric.MAXIMIZE,
                 regex=".*\\[[0-9]+\\].*#011validation-map:(\\S+)"),
        m.Metric(name="validation:merror",
                 direction=m.Metric.MINIMIZE,
                 regex=".*\\[[0-9]+\\].*#011validation-merror:(\\S+)"),
        m.Metric(name="validation:mlogloss",
                 direction=m.Metric.MINIMIZE,
                 regex=".*\\[[0-9]+\\].*#011validation-mlogloss:(\\S+)"),
        m.Metric(name="validation:mse",
                 direction=m.Metric.MINIMIZE,
                 regex=".*\\[[0-9]+\\].*#011validation-mse:(\\S+)"),
        m.Metric(name="validation:ndcg",
                 direction=m.Metric.MAXIMIZE,
                 regex=".*\\[[0-9]+\\].*#011validation-ndcg:(\\S+)"),
        m.Metric(name="validation:rmse",
                 direction=m.Metric.MINIMIZE,
                 regex=".*\\[[0-9]+\\].*#011validation-rmse:(\\S+)"))
