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
from __future__ import absolute_import

import numpy as np
import xgboost as xgb


from sagemaker_xgboost_container.algorithm_mode import train_utils


def test_get_union_metrics():
    a = ['metric_1', 'metric_2']
    b = ['metric_1', 'metric_3']

    union = train_utils.get_union_metrics(a, b)
    assert len(union) == 3
    for metric in union:
        assert metric in ['metric_1', 'metric_2', 'metric_3']


def test_get_eval_metrics_and_feval():
    test_objective = 'validation:logloss'
    test_evals = ['accuracy', 'rmse']

    test_eval_metrics, test_configured_eval = train_utils.get_eval_metrics_and_feval(test_objective, test_evals)

    assert len(test_eval_metrics) == 2
    for metric in test_eval_metrics:
        assert metric in ['logloss', 'rmse']

    binary_train_data = np.random.rand(10, 2)
    binary_train_label = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    binary_dtrain = xgb.DMatrix(binary_train_data, label=binary_train_label)
    binary_preds = np.ones(10)

    assert ('accuracy', .5) == test_configured_eval(binary_preds, binary_dtrain)[0]


def test_get_latest_checkpoint():
    checkpoint_files1 = ['xgboost-checkpoint.1', 'xgboost-checkpoint.2', 'tmp_checkpoint.1']
    checkpoint_files2 = ['tmp_checkpoint']
    checkpoint_files3 = []

    file_name = train_utils.get_latest_checkpoint(checkpoint_files1)
    assert file_name == 'xgboost-checkpoint.2'

    file_name = train_utils.get_latest_checkpoint(checkpoint_files2)
    assert file_name is None

    file_name = train_utils.get_latest_checkpoint(checkpoint_files3)
    assert file_name is None
