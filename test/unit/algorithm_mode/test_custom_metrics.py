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
import xgboost as xgb

from sagemaker_xgboost_container.metrics.custom_metrics import accuracy, f1, mse


binary_train_data = np.random.rand(10, 2)
binary_train_label = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
binary_dtrain = xgb.DMatrix(binary_train_data, label=binary_train_label)
binary_preds = np.ones(10)


def test_binary_accuracy():
    accuracy_name, accuracy_result = accuracy(binary_preds, binary_dtrain)
    assert accuracy_name == 'accuracy'
    assert accuracy_result == .5


def test_binary_f1():
    f1_score_name, f1_score_result = f1(binary_preds, binary_dtrain)
    assert f1_score_name == 'f1'
    assert f1_score_result == 1/3


def test_mse():
    mse_score_name, mse_score_result = mse(binary_preds, binary_dtrain)
    assert mse_score_name == 'mse'
    assert mse_score_result == .5


multiclass_train_data = np.random.rand(10, 2)
multiclass_train_label = np.array([0, 0, 1, 1, 1, 1, 1, 2, 2, 2])
multiclass_dtrain = xgb.DMatrix(multiclass_train_data, label=multiclass_train_label)
multiclass_preds = np.ones(10)
multiclass_preds_softprob = np.asarray([[0.8, 0.1, 0.1]] * 10)


def test_multiclass_accuracy():
    accuracy_name, accuracy_result = accuracy(multiclass_preds, multiclass_dtrain)
    assert accuracy_name == 'accuracy'
    assert accuracy_result == .5


def test_multiclass_accuracy_softprob():
    accuracy_name, accuracy_result = accuracy(multiclass_preds_softprob, multiclass_dtrain)
    assert accuracy_name == 'accuracy'
    assert accuracy_result == .2


def test_multiclass_f1():
    f1_score_name, f1_score_result = f1(multiclass_preds, multiclass_dtrain)
    assert f1_score_name == 'f1'
    assert f1_score_result == 2/9
