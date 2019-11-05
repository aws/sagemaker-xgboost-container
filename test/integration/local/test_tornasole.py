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
import os

from tornasole.trials import create_trial
from tornasole.xgboost.hook import DEFAULT_SAVE_CONFIG_INTERVAL

from test.utils import local_mode, test_utils

path = os.path.dirname(os.path.realpath(__file__))
source_dir = os.path.join(path, '..', '..', 'resources', 'tornasole')
data_dir = os.path.join(source_dir, 'data')


def get_abalone_hyperparameters():

    hyperparameters = {
        "max_depth": 5,
        "eta": 0.2,
        "gamma": 4,
        "min_child_weight": 6,
        "subsample": 0.7,
        "silent": 0,
        "num_round": 20,
    }

    return hyperparameters


def test_tornasole_algorithm_mode_negative(docker_image, opt_ml):

    customer_script = None  # algorithm mode
    hyperparameters = get_abalone_hyperparameters()

    local_mode.train(customer_script, data_dir, docker_image, opt_ml,
                     hyperparameters=hyperparameters)

    assert not local_mode.file_exists(opt_ml, "output/failure")

    output_dir = os.path.join(opt_ml, 'algo-1', 'output')
    assert "tensors" not in os.listdir(output_dir)


def test_tornasole_algorithm_mode_default_positive_no_params(docker_image, opt_ml):

    customer_script = None  # algorithm mode
    hyperparameters = get_abalone_hyperparameters()

    debughookconfig = {
        "S3OutputPath": "s3://bucket/prefix",
        "LocalPath": "/opt/ml/output/tensors",
    }

    local_mode.train(customer_script, data_dir, docker_image, opt_ml,
                     hyperparameters=hyperparameters, debughookconfig=debughookconfig)

    assert not local_mode.file_exists(opt_ml, "output/failure")

    tensors_dir = os.path.join(opt_ml, 'algo-1', 'output', 'tensors')
    trial = create_trial(tensors_dir)
    assert trial.tensors() == ["train-rmse", "validation-rmse"]
    assert trial.available_steps() == list(range(0, 20, DEFAULT_SAVE_CONFIG_INTERVAL))


def test_tornasole_algorithm_mode_default_positive_null_params(docker_image, opt_ml):

    customer_script = None  # algorithm mode
    hyperparameters = get_abalone_hyperparameters()

    debughookconfig = {
        "S3OutputPath": "s3://bucket/prefix",
        "LocalPath": "/opt/ml/output/tensors",
        "HookParameters": None,
        "CollectionConfigurations": None
    }

    local_mode.train(customer_script, data_dir, docker_image, opt_ml,
                     hyperparameters=hyperparameters, debughookconfig=debughookconfig)

    assert not local_mode.file_exists(opt_ml, "output/failure")

    tensors_dir = os.path.join(opt_ml, 'algo-1', 'output', 'tensors')
    trial = create_trial(tensors_dir)
    assert trial.tensors() == ["train-rmse", "validation-rmse"]
    assert trial.available_steps() == list(range(0, 20, DEFAULT_SAVE_CONFIG_INTERVAL))


def test_tornasole_algorithm_mode_hook_parameters(docker_image, opt_ml):

    customer_script = None  # algorithm mode
    hyperparameters = get_abalone_hyperparameters()

    debughookconfig = {
        "S3OutputPath": "s3://bucket/prefix",
        "LocalPath": "/opt/ml/output/tensors",
        "HookParameters": {
            "save_interval": 7,
            "save_steps": "0,1,2,3"
        },
        "CollectionConfigurations": None
    }

    local_mode.train(customer_script, data_dir, docker_image, opt_ml,
                     hyperparameters=hyperparameters, debughookconfig=debughookconfig)

    assert not local_mode.file_exists(opt_ml, "output/failure")

    tensors_dir = os.path.join(opt_ml, 'algo-1', 'output', 'tensors')
    trial = create_trial(tensors_dir)
    assert trial.tensors() == ["train-rmse", "validation-rmse"]
    assert trial.available_steps() == [0, 1, 2, 3, 7, 14]


def test_tornasole_algorithm_mode_collection_configurations(docker_image, opt_ml):

    customer_script = None  # algorithm mode
    hyperparameters = get_abalone_hyperparameters()

    debughookconfig = {
        "S3OutputPath": "s3://bucket/prefix",
        "LocalPath": "/opt/ml/output/tensors",
        "HookParameters": None,
        "CollectionConfiguration": [
            {"CollectionName": "hyperparameters"},
            {"CollectionName": "average_shap"},
            {"CollectionName": "predictions"}
        ]
    }

    local_mode.train(customer_script, data_dir, docker_image, opt_ml,
                     hyperparameters=hyperparameters, debughookconfig=debughookconfig)

    assert not local_mode.file_exists(opt_ml, "output/failure")

    tensors_dir = os.path.join(opt_ml, 'algo-1', 'output', 'tensors')
    trial = create_trial(tensors_dir)
    tensors = trial.tensors()
    assert any("average_shap" in tensor for tensor in tensors)
    expected_hp = ["hyperparameters/{}".format(hp) for hp in hyperparameters.keys()]
    assert all(hp in tensors for hp in expected_hp)
    assert "predictions" in tensors


def test_tornasole_script_mode_single_machine(docker_image, opt_ml):

    customer_script = "xgboost_abalone_basic_hook_demo.py"
    hyperparameters = get_abalone_hyperparameters()

    local_mode.train(customer_script, data_dir, docker_image, opt_ml,
                     hyperparameters=hyperparameters, source_dir=source_dir)

    assert not local_mode.file_exists(opt_ml, 'output/failure'), 'Failure happened'

    tensors_dir = os.path.join(opt_ml, 'algo-1', 'output', 'tensors')
    trial = create_trial(tensors_dir)
    assert trial.tensors() == ["train-rmse", "validation-rmse"]
    assert trial.available_steps() == list(range(0, 20))
