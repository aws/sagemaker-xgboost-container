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
        test_hp = {
            'eval_metric': 'auc'}

        auc_invalid_objectives = [
            'count:poisson',
            'reg:gamma',
            'reg:logistic',
            'reg:squarederror',
            'reg:tweedie',
            'multi:softmax',
            'multi:softprob',
            'survival:cox']

        for invalid_objective in auc_invalid_objectives:
            test_hp['objective'] = invalid_objective

            with self.assertRaises(exc.UserError):
                hyperparameters.validate(test_hp)
