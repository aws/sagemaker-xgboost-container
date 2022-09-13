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

XGB_MAXIMIZE_METRICS = [
    "accuracy",
    "auc",
    "aucpr",
    "balanced_accuracy",
    "f1",
    "f1_binary",
    "f1_macro",
    "map",
    "ndcg",
    "precision",
    "r2",
    "recall",
    "precision_macro",
    "precision_micro",
    "recall_macro",
    "recall_micro",
]

XGB_MINIMIZE_METRICS = [
    "aft-nloglik",
    "cox-nloglik",
    "error",
    "gamma-deviance",
    "gamma-nloglik",
    "interval-regression-accuracy",
    "logloss",
    "mae",
    "mape",
    "merror",
    "mlogloss",
    "mphe",
    "mse",
    "poisson-nloglik",
    "rmse",
    "rmsle",
    "tweedie-nloglik",
]

LOGISTIC_REGRESSION_LABEL_RANGE_ERROR = "label must be in [0,1] for logistic regression"
MULTI_CLASS_LABEL_RANGE_ERROR = "label must be in [0, num_class)"
MULTI_CLASS_F1_BINARY_ERROR = "Target is multiclass but average='binary'"
FEATURE_MISMATCH_ERROR = "feature_names mismatch"
LABEL_PREDICTION_SIZE_MISMATCH = "Check failed: preds.size() == info.labels_.size()"
ONLY_POS_OR_NEG_SAMPLES = "Check failed: !auc_error AUC: the dataset only contains pos or neg samples"
BASE_SCORE_RANGE_ERROR = (
    "Check failed: base_score > 0.0f && base_score < 1.0f base_score must be in (0,1) " "for logistic loss"
)
POISSON_REGRESSION_ERROR = "Check failed: label_correct PoissonRegression: label must be nonnegative"
TWEEDIE_REGRESSION_ERROR = "Check failed: label_correct TweedieRegression: label must be nonnegative"
REG_LAMBDA_ERROR = "Parameter reg_lambda should be greater equal to 0"

CUSTOMER_ERRORS = [
    LOGISTIC_REGRESSION_LABEL_RANGE_ERROR,
    MULTI_CLASS_LABEL_RANGE_ERROR,
    MULTI_CLASS_F1_BINARY_ERROR,
    FEATURE_MISMATCH_ERROR,
    LABEL_PREDICTION_SIZE_MISMATCH,
    ONLY_POS_OR_NEG_SAMPLES,
    BASE_SCORE_RANGE_ERROR,
    POISSON_REGRESSION_ERROR,
    TWEEDIE_REGRESSION_ERROR,
    REG_LAMBDA_ERROR,
]

_SEPARATOR = ":"
TRAIN_CHANNEL = "train"
VAL_CHANNEL = "validation"

# xgboost objective learning tasks
# https://xgboost.readthedocs.io/en/release_1.0.0/parameter.html#learning-task-parameters
REG_SQUAREDERR = "reg:squarederror"
REG_LOG = "reg:logistic"
REG_GAMMA = "reg:gamma"
REG_TWEEDIE = "reg:tweedie"
BINARY_LOG = "binary:logistic"
BINARY_LOGRAW = "binary:logitraw"
BINARY_HINGE = "binary:hinge"
MULTI_SOFTMAX = "multi:softmax"
MULTI_SOFTPROB = "multi:softprob"
