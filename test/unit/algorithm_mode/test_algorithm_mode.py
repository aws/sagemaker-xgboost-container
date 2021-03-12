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
import mock
import unittest

from sagemaker_xgboost_container.algorithm_mode import channel_validation as cv
from sagemaker_xgboost_container.algorithm_mode import hyperparameter_validation as hpv
from sagemaker_xgboost_container.algorithm_mode import metrics as metrics_mod


def basic_training_environment():
    channel_input_dirs = {
        "train": {
            "ContentType": "text/csv",
            "TrainingInputMode": "File",
            "S3DistributionType": "FullyReplicated"}}
    return mock.Mock(channel_input_dirs=channel_input_dirs,
                     hyperparameters={"num_round": "100"})


class TestAlgorithmModeHyperparameters(unittest.TestCase):
    def setUp(self):
        self.metrics = mock.Mock(names=[])
        self.training_environment = basic_training_environment()

    def test_hyperparameters(self):
        hyperparameters = {
            "max_depth": "5",
            "eta": "0.2",
            "gamma": "4",
            "min_child_weight": "6",
            "subsample": "0.8",
            "objective": "binary:logistic",
            "num_round": "100"}
        hps = hpv.initialize(self.metrics)
        hps.validate(hyperparameters)

    def test_hyperparameters2(self):
        hyperparameters = {
            "eval_metric": "auc",
            "objective": "binary:logistic",
            "num_round": "100",
            "rate_drop": "0.3",
            "tweedie_variance_power": "1.4"
        }
        hps = hpv.initialize(self.metrics)
        hps.validate(hyperparameters)

    def test_hyperparameters3(self):
        hyperparameters = {
            "max_depth": "5",
            "eta": "0.2",
            "gamma": "4",
            "min_child_weight": "6",
            "subsample": "0.7",
            "objective": "reg:squarederror",
            "num_round": "50"
        }
        hps = hpv.initialize(self.metrics)
        hps.validate(hyperparameters)

    def test_hyperparameters4(self):
        hyperparameters = {
            "max_depth": "5",
            "eta": "0.2",
            "gamma": "4",
            "min_child_weight": "6",
            "objective": "multi:softmax",
            "num_class": "10",
            "num_round": "10"
        }
        hps = hpv.initialize(self.metrics)
        hps.validate(hyperparameters)

    def test_hyperparameters5(self):
        hyperparameters = {
            "max_depth": "5",
            "eta": "0.2",
            "gamma": "4",
            "min_child_weight": "6",
            "tree_method": "exact",
            "objective": "multi:softmax",
            "num_class": "10",
            "num_round": "10",
            "monotone_constraints": "(1,0)",
            "interaction_constraints": "[[1,2,4],[3,5]]"
        }
        hps = hpv.initialize(self.metrics)
        hps.validate(hyperparameters)

    def test_hyperparameters6(self):
        hyperparameters = {
            "max_depth": "5",
            "eta": "0.2",
            "gamma": "4",
            "min_child_weight": "6",
            "tree_method": "approx",
            "objective": "multi:softmax",
            "num_class": "10",
            "num_round": "10",
            "interaction_constraints": "[[1,2,4],[3,5]]"
        }
        hps = hpv.initialize(self.metrics)
        hps.validate(hyperparameters)

    def test_hyperparameters7(self):
        hyperparameters = {
            "max_depth": "5",
            "learning_rate": "0.2",
            "gamma": "4",
            "min_child_weight": "6",
            "tree_method": "approx",
            "objective": "multi:softmax",
            "num_class": "10",
            "num_round": "10",
            "interaction_constraints": "[[1,2,4],[3,5]]"
        }
        hps = hpv.initialize(self.metrics)
        hps.validate(hyperparameters)

    def test_hyperparameters8(self):
        hyperparameters = {
            "max_depth": "5",
            "eta": "0.2",
            "min_split_loss": "4",
            "min_child_weight": "6",
            "tree_method": "approx",
            "objective": "multi:softmax",
            "num_class": "10",
            "num_round": "10",
            "interaction_constraints": "[[1,2,4],[3,5]]"
        }
        hps = hpv.initialize(self.metrics)
        hps.validate(hyperparameters)

    def test_hyperparameters9(self):
        hyperparameters = {
            "max_depth": "5",
            "eta": "0.2",
            "gamma": "4",
            "reg_lambda": "10",
            "min_child_weight": "6",
            "objective": "multi:softmax",
            "num_class": "10",
            "num_round": "10"
        }
        hps = hpv.initialize(self.metrics)
        hps.validate(hyperparameters)

    def test_hyperparameters10(self):
        hyperparameters = {
            "max_depth": "5",
            "eta": "0.2",
            "gamma": "4",
            "reg_alpha": "10",
            "min_child_weight": "6",
            "objective": "multi:softmax",
            "num_class": "10",
            "num_round": "10"
        }
        hps = hpv.initialize(self.metrics)
        hps.validate(hyperparameters)

    def test_hyperparameters11(self):
        hyperparameters = {
            "max_depth": "5",
            "eta": "0.2",
            "gamma": "4",
            "reg_alpha": "10",
            "min_child_weight": "6",
            "objective": "multi:softmax",
            "num_class": "10",
            "num_round": "10",
            "_kfold": "5"
        }
        hps = hpv.initialize(self.metrics)
        hps.validate(hyperparameters)


class TestAlgorithmModeChannels(unittest.TestCase):
    def setUp(self):
        self.training_environment = basic_training_environment()

    def test_channels(self):
        input_data_config = {
            "train": {
                "ContentType": "csv",
                "TrainingInputMode": "File",
                "S3DistributionType": "FullyReplicated",
                "RecordWrapperType": "None"
            },
            "validation": {
                "ContentType": "csv",
                "TrainingInputMode": "File",
                "S3DistributionType": "FullyReplicated",
                "RecordWrapperType": "None"
            }
        }
        cs = cv.initialize()
        cs.validate(input_data_config)

    def test_channels_libsvm(self):
        input_data_config = {
            "train": {
                "ContentType": "csv",
                "TrainingInputMode": "File",
                "S3DistributionType": "FullyReplicated",
                "RecordWrapperType": "None"
            },
            "validation": {
                "ContentType": "csv",
                "TrainingInputMode": "File",
                "S3DistributionType": "FullyReplicated",
                "RecordWrapperType": "None"
            }
        }
        cs = cv.initialize()
        cs.validate(input_data_config)


class TestAlgorithmModeMetrics(unittest.TestCase):
    def setUp(self):
        self.training_environment = basic_training_environment()

    def test_metrics(self):
        metrics_mod.initialize()
