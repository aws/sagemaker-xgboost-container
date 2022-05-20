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
from math import log, sqrt
from sagemaker_xgboost_container.metrics.custom_metrics import accuracy, f1, mse, r2, f1_binary, f1_macro, \
    precision_macro, precision_micro, recall_macro, recall_micro, mae, rmse, balanced_accuracy, \
    precision, recall


binary_train_data = np.random.rand(10, 2)
binary_train_label = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
binary_dtrain = xgb.DMatrix(binary_train_data, label=binary_train_label)
# log(x/(1-x)) is the inverse function of sigmoid
binary_preds = np.asarray([log(0.7/0.3) - 0.5] * 10)
binary_preds_logistic = np.asarray([[log(0.1/0.9) - 0.5, log(0.9/0.1) - 0.5]] * 10)


def test_binary_accuracy():
    accuracy_name, accuracy_result = accuracy(binary_preds, binary_dtrain)
    assert accuracy_name == 'accuracy'
    assert accuracy_result == .5

def test_binary_balanced_accuracy():
    bal_accuracy_name, bal_accuracy_result = balanced_accuracy(binary_preds, binary_dtrain)
    assert bal_accuracy_name == 'balanced_accuracy'
    assert bal_accuracy_result == .5


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


def test_binary_f1_binary():
    f1_score_name, f1_score_result = f1_binary(binary_preds, binary_dtrain)
    assert f1_score_name == 'f1_binary'
    assert f1_score_result == 2/3


def test_binary_f1_binary_logistic():
    f1_score_name, f1_score_result = f1_binary(binary_preds_logistic, binary_dtrain)
    assert f1_score_name == 'f1_binary'
    assert f1_score_result == 2/3

def test_binary_precision():
    precision_score_name, precision_score_result = precision(binary_preds, binary_dtrain)
    assert precision_score_name == 'precision'
    assert precision_score_result == .5

def test_binary_recall():
    recall_score_name, recall_score_result = recall(binary_preds, binary_dtrain)
    assert recall_score_name == 'recall'
    assert recall_score_result == 1


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

def test_multiclass_balanced_accuracy():
    bal_accuracy_name, bal_accuracy_result = balanced_accuracy(multiclass_preds, multiclass_dtrain)
    assert bal_accuracy_name == 'balanced_accuracy'
    assert balanced_accuracy_result == .5

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


def test_multiclass_f1_macro():
    f1_score_name, f1_score_result = f1_macro(multiclass_preds, multiclass_dtrain)
    assert f1_score_name == 'f1_macro'
    assert f1_score_result == 2/9


def test_multiclass_precision_macro():
    precision_macro_name, precision_macro_result = precision_macro(multiclass_preds, multiclass_dtrain)
    assert precision_macro_name == 'precision_macro'
    assert precision_macro_result == 1/6


def test_multiclass_precision_micro():
    precision_micro_name, precision_micro_result = precision_micro(multiclass_preds, multiclass_dtrain)
    assert precision_micro_name == 'precision_micro'
    assert precision_micro_result == 1/2


def test_multiclass_recall_macro():
    recall_macro_name, recall_macro_result = recall_macro(multiclass_preds, multiclass_dtrain)
    assert recall_macro_name == 'recall_macro'
    assert recall_macro_result == 1/3


def test_multiclass_recall_micro():
    recall_micro_name, recall_micro_result = recall_micro(multiclass_preds, multiclass_dtrain)
    assert recall_micro_name == 'recall_micro'
    assert recall_micro_result == 1/2


def test_multiclass_f1_macro_softprob():
    f1_score_name, f1_score_result = f1_macro(multiclass_preds_softprob, multiclass_dtrain)
    assert f1_score_name == 'f1_macro'
    assert f1_score_result == 1/15


regression_train_data = np.random.rand(10, 2)
regression_train_label = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
regression_dtrain = xgb.DMatrix(regression_train_data, label=regression_train_label)
regression_preds = np.ones(10)


def test_mse():
    mse_score_name, mse_score_result = mse(regression_preds, regression_dtrain)
    assert mse_score_name == 'mse'
    assert mse_score_result == .5


def test_r2():
    r2_score_name, r2_score_result = r2(regression_preds, regression_dtrain)
    assert r2_score_name == 'r2'
    assert r2_score_result == -1

def test_rmse():
    rmse_score_name, rmse_score_result = rmse(regression_preds, regression_dtrain)
    assert rmse_score_name == 'rmse'
    assert rmse_score_result == sqrt(0.5)

def test_mae():
    mae_score_name, mae_score_result = mae(regression_preds, regression_dtrain)
    assert mae_score_name == 'mae'
    assert mae_score_result == .5
