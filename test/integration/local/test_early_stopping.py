# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import os
from test.utils import local_mode

path = os.path.dirname(os.path.realpath(__file__))
early_stopping_path = os.path.join(path, "..", "..", "resources", "early_stopping")
data_dir = os.path.join(early_stopping_path, "data")


def get_default_hyperparameters(num_round=100):
    hyperparameters = {
        "max_depth": "5",
        "eta": "0.2",
        "gamma": "4",
        "min_child_weight": "6",
        "subsample": "0.7",
        "num_round": str(num_round),
    }
    return hyperparameters


def test_xgboost_training_single_machine_with_early_stopping(docker_image, opt_ml):
    hyperparameters = get_default_hyperparameters(100000)
    hyperparameters["save_model_on_termination"] = "true"

    local_mode.train(
        False, data_dir, docker_image, opt_ml, hyperparameters=hyperparameters, early_stopping=True, train_time=10
    )

    assert local_mode.file_exists(opt_ml, "model/xgboost-model"), "Model not saved"


def test_xgboost_training_single_machine_without_early_stopping(docker_image, opt_ml):
    hyperparameters = get_default_hyperparameters(100000)
    hyperparameters["save_model_on_termination"] = "false"

    local_mode.train(
        False, data_dir, docker_image, opt_ml, hyperparameters=hyperparameters, early_stopping=True, train_time=10
    )

    assert not local_mode.file_exists(opt_ml, "model/xgboost-model"), "Model saved"


# def test_xgboost_training_multiple_machines_with_early_stopping(docker_image, opt_ml):
#     hyperparameters = get_default_hyperparameters(100000)
#     hyperparameters["save_model_on_termination"] = "true"

#     local_mode.train(
#         False, data_dir, docker_image, opt_ml, hyperparameters=hyperparameters, cluster_size=2, early_stopping=True
#     )

#     host1 = local_mode.file_exists(opt_ml, "model/xgboost-model", "algo-1")
#     host2 = local_mode.file_exists(opt_ml, "model/xgboost-model", "algo-2")
#     assert host1 or host2, "Model not saved on any host"
#     assert not (host1 and host2), "Model saved on both hosts"


def test_xgboost_training_multiple_machines_without_early_stopping(docker_image, opt_ml):
    hyperparameters = get_default_hyperparameters(100000)
    hyperparameters["save_model_on_termination"] = "false"

    local_mode.train(
        False, data_dir, docker_image, opt_ml, hyperparameters=hyperparameters, cluster_size=2, early_stopping=True
    )

    host1 = local_mode.file_exists(opt_ml, "model/xgboost-model", "algo-1")
    host2 = local_mode.file_exists(opt_ml, "model/xgboost-model", "algo-2")
    assert not (host1 or host2), "Model saved on some host"
