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
import os
import tempfile
import shutil
import math

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

    test_eval_metrics, test_configured_eval, tuning_metric = train_utils.get_eval_metrics_and_feval(test_objective,
                                                                                                    test_evals)

    assert len(test_eval_metrics) == 1
    for metric in test_eval_metrics:
        assert metric in ['logloss']

    binary_train_data = np.random.rand(10, 2)
    binary_train_label = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    binary_dtrain = xgb.DMatrix(binary_train_data, label=binary_train_label)
    binary_preds = np.ones(10)

    custom_metric_results = test_configured_eval(binary_preds, binary_dtrain)
    custom_metric_results.sort()

    assert 2 == len(custom_metric_results)
    assert ('accuracy', .5) == custom_metrics_results[0]
    assert ('rmse', math.sqrt(0.5)) == custom_metrics_results[1]

def test_cleanup_dir():
    def setup(file_names):
        test_dir = tempfile.mkdtemp()
        for file_name in file_names:
            test_path = os.path.join(test_dir, file_name)
            with open(test_path, 'w'):
                pass

        return test_dir

    def tearDown(dir):
        shutil.rmtree(dir)

    # Test 1: Check if 'xgboost-model' is present after cleanup
    model_name = "xgboost-model"
    file_names = ['tmp1', 'tmp2', 'xgboost-model']
    test_dir = setup(file_names)

    train_utils.cleanup_dir(test_dir, model_name)
    files = os.listdir(test_dir)

    assert len(files) == 1
    assert files[0] == model_name

    tearDown(test_dir)

    # Test 2: Check if directory is empty after cleanup
    file_names = ['tmp1', 'tmp2']
    test_dir = setup(file_names)

    train_utils.cleanup_dir(test_dir, model_name)
    files = os.listdir(test_dir)

    assert len(files) == 0

    tearDown(test_dir)

    # Test 3: Check if directory is empty after cleanup
    file_names = []
    test_dir = setup(file_names)

    train_utils.cleanup_dir(test_dir, model_name)
    files = os.listdir(test_dir)

    assert len(files) == 0

    tearDown(test_dir)
