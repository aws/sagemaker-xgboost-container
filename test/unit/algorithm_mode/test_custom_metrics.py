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
from math import log
from sagemaker_xgboost_container.metrics.custom_metrics import accuracy, f1, mse


binary_train_data = np.random.rand(10, 2)
binary_train_label = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
binary_dtrain = xgb.DMatrix(binary_train_data, label=binary_train_label)
# log(x/(1-x)) is the inverse function of sigmoid
binary_preds = [log(0.7/0.3) - 0.5] * 10
binary_preds_logistic = np.asarray([[log(0.1/0.9) - 0.5, log(0.9/0.1) - 0.5]] * 10)


def test_binary_accuracy():
    accuracy_name, accuracy_result = accuracy(binary_preds, binary_dtrain)
    assert accuracy_name == 'accuracy'
    assert accuracy_result == .5


def test_binary_accuracy_logistic():
    accuracy_name, accuracy_result = accuracy(binary_preds_logistic, binary_dtrain)
    assert accuracy_name == 'accuracy'
    assert accuracy_result == .5


def test_binary_f1():
    f1_score_name, f1_score_result = f1(binary_preds, binary_dtrain)
    assert f1_score_name == 'f1'
    assert f1_score_result == 1/3


def test_binary_f1_logistic():
    f1_score_name, f1_score_result = f1(binary_preds_logistic, binary_dtrain)
    assert f1_score_name == 'f1'
    assert f1_score_result == 1/3


multiclass_train_data = np.random.rand(10, 2)
multiclass_train_label = np.array([0, 0, 1, 1, 1, 1, 1, 2, 2, 2])
multiclass_dtrain = xgb.DMatrix(multiclass_train_data, label=multiclass_train_label)
multiclass_preds = np.ones(10)
multiclass_preds_softprob = np.asarray([[0.9, 0.05, 0.05],
                                        [0.4, 0.5, 0.1],
                                        [0.2, 0.1, 0.7],
                                        [0.8, 0.1, 0.1],
                                        [0.6, 0.2, 0.2],
                                        [0.9, 0.08, 0.02],
                                        [0.4, 0.3, 0.3],
                                        [0.5, 0.25, 0.25],
                                        [0.8, 0.1, 0.1],
                                        [0.6, 0.2, 0.2]])


def test_multiclass_accuracy():
    accuracy_name, accuracy_result = accuracy(multiclass_preds, multiclass_dtrain)
    assert accuracy_name == 'accuracy'
    assert accuracy_result == .5


def test_multiclass_accuracy_softprob():
    accuracy_name, accuracy_result = accuracy(multiclass_preds_softprob, multiclass_dtrain)
    assert accuracy_name == 'accuracy'
    assert accuracy_result == .1


def test_multiclass_f1():
    f1_score_name, f1_score_result = f1(multiclass_preds, multiclass_dtrain)
    assert f1_score_name == 'f1'
    assert f1_score_result == 2/9


def test_multiclass_f1_softprob():
    f1_score_name, f1_score_result = f1(multiclass_preds_softprob, multiclass_dtrain)
    assert f1_score_name == 'f1'
    assert f1_score_result == 1/15
