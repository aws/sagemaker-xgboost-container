import numpy as np
import xgboost as xgb

from sagemaker_xgboost_container.metrics.custom_metrics import accuracy, f1


train_data = np.random.rand(10, 2)
train_label = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
dtrain = xgb.DMatrix(train_data, label=train_label)
preds = np.ones(10)


def test_accuracy():
    accuracy_name, accuracy_result = accuracy(preds, dtrain)
    assert accuracy_name == 'accuracy'
    assert accuracy_result == .5


def test_f1():
    f1_score_name, f1_score_result = f1(preds, dtrain)
    assert f1_score_name == 'f1'
