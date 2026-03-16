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

import unittest

from sagemaker_algorithm_toolkit import exceptions as exc
from sagemaker_xgboost_container.algorithm_mode import hyperparameter_validation as hpv
from sagemaker_xgboost_container.algorithm_mode import metrics as metrics_mod

metrics = metrics_mod.initialize()
hyperparameters = hpv.initialize(metrics)


class TestHyperparameterValidation(unittest.TestCase):
    def test_auc_invalid_objective(self):
        test_hp = {"eval_metric": "auc"}

        auc_invalid_objectives = [
            "count:poisson",
            "reg:gamma",
            "reg:logistic",
            "reg:squarederror",
            "reg:tweedie",
            "multi:softmax",
            "multi:softprob",
            "survival:cox",
        ]

        for invalid_objective in auc_invalid_objectives:
            test_hp["objective"] = invalid_objective

            with self.assertRaises(exc.UserError):
                hyperparameters.validate(test_hp)

    def test_verbosity(self):
        test_hp = {"num_round": "1", "verbosity": "0"}

        assert hyperparameters.validate(test_hp)

        test_hp2 = {"num_round": "1", "verbosity": "3"}

        assert hyperparameters.validate(test_hp2)

        test_hp3 = {"num_round": "1", "verbosity": "4"}

        with self.assertRaises(exc.UserError):
            hyperparameters.validate(test_hp3)

    def test_num_parallel_tree(self):
        test_hp = {"num_round": "5", "num_parallel_tree": "10"}

        assert hyperparameters.validate(test_hp)

        test_hp2 = {"num_round": "5", "num_parallel_tree": "-1"}

        with self.assertRaises(exc.UserError):
            hyperparameters.validate(test_hp2)

        test_hp3 = {"num_round": "5", "num_parallel_tree": "0"}

        with self.assertRaises(exc.UserError):
            hyperparameters.validate(test_hp3)

    def test_save_model_on_termination(self):
        test_hp1 = {"num_round": "5", "save_model_on_termination": "true"}

        assert hyperparameters.validate(test_hp1)

        test_hp2 = {"num_round": "5", "save_model_on_termination": "false"}

        assert hyperparameters.validate(test_hp2)

        test_hp3 = {"num_round": "5", "save_model_on_termination": "incorrect"}

        with self.assertRaises(exc.UserError):
            hyperparameters.validate(test_hp3)

    def test_survival_analysis(self):
        test_hp1 = {
            "num_round": "1",
            "eval_metric": "aft-nloglik",
            "objective": "reg:squarederror",
        }
        with self.assertRaises(exc.UserError):
            hyperparameters.validate(test_hp1)

        test_hp2 = {
            "num_round": "1",
            "eval_metric": "aft-nloglik",
            "objective": "survival:aft",
        }
        assert hyperparameters.validate(test_hp2)

    def test_learning_to_rank_params_valid_with_rank_objective(self):
        """Learning-to-rank params are valid when objective is rank:ndcg, rank:map, or rank:pairwise."""
        for objective in ("rank:ndcg", "rank:map", "rank:pairwise"):
            test_hp = {
                "num_round": "10",
                "objective": objective,
                "lambdarank_pair_method": "topk",
                "lambdarank_num_pair_per_sample": "6",
                "lambdarank_normalization": "true",
                "lambdarank_score_normalization": "true",
                "lambdarank_unbiased": "false",
                "lambdarank_bias_norm": "2.0",
                "ndcg_exp_gain": "true",
            }
            assert hyperparameters.validate(test_hp), "Failed for objective={}".format(objective)

        test_hp_mean = {
            "num_round": "10",
            "objective": "rank:ndcg",
            "lambdarank_pair_method": "mean",
            "lambdarank_num_pair_per_sample": "5",
        }
        assert hyperparameters.validate(test_hp_mean)

    def test_learning_to_rank_params_invalid_when_objective_missing(self):
        """Learning-to-rank params raise UserError when objective is not set."""
        test_hp = {"num_round": "10", "lambdarank_pair_method": "topk"}
        with self.assertRaises(exc.UserError):
            hyperparameters.validate(test_hp)

    def test_learning_to_rank_params_invalid_with_non_rank_objective(self):
        """Learning-to-rank params raise UserError when objective is not a rank objective."""
        ltr_params = {
            "lambdarank_pair_method": "topk",
            "lambdarank_num_pair_per_sample": "6",
            "lambdarank_normalization": "true",
            "lambdarank_score_normalization": "true",
            "lambdarank_unbiased": "false",
            "lambdarank_bias_norm": "2.0",
            "ndcg_exp_gain": "true",
        }
        non_rank_objectives = ["reg:squarederror", "binary:logistic", "multi:softmax"]

        for param_name, param_value in ltr_params.items():
            for objective in non_rank_objectives:
                test_hp = {"num_round": "10", "objective": objective, param_name: param_value}
                with self.assertRaises(exc.UserError):
                    hyperparameters.validate(test_hp)

    def test_learning_to_rank_params_invalid_values(self):
        """Invalid values for LTR params raise UserError."""
        base_hp = {"num_round": "10", "objective": "rank:ndcg"}

        with self.assertRaises(exc.UserError):
            hyperparameters.validate({**base_hp, "lambdarank_pair_method": "invalid"})

        with self.assertRaises(exc.UserError):
            hyperparameters.validate({**base_hp, "lambdarank_num_pair_per_sample": "0"})

        with self.assertRaises(exc.UserError):
            hyperparameters.validate({**base_hp, "lambdarank_bias_norm": "-0.1"})

        with self.assertRaises(exc.UserError):
            hyperparameters.validate({**base_hp, "ndcg_exp_gain": "invalid"})
