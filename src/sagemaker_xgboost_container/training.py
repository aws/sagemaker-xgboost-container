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

import json
import logging
import os
import sys

from sagemaker_containers import _env
import sagemaker_containers.beta.framework as framework
from sagemaker_xgboost_container.algorithm_mode.train import sagemaker_train
from sagemaker_xgboost_container.data_utils import check_data_redundancy
from sagemaker_xgboost_container.constants import sm_env_constants

logger = logging.getLogger(__name__)


def run_algorithm_mode():
    """Run training in algorithm mode, which does not require a user entry point.

    This parses the following environ elements for training:

        'SM_INPUT_TRAINING_CONFIG_FILE'
        'SM_INPUT_DATA_CONFIG_FILE'
        'SM_CHANNEL_TRAIN'
        'SM_CHANNEL_VALIDATION'
        'SM_HOSTS'
        'SM_CURRENT_HOST'
        'SM_MODEL_DIR'
        'SM_CHECKPOINT_CONFIG_FILE'

    """
    # TODO: replace with CSDK constants in sagemaker_containers._env
    with open(os.getenv(sm_env_constants.SM_INPUT_TRAINING_CONFIG_FILE), "r") as f:
        train_config = json.load(f)
    with open(os.getenv(sm_env_constants.SM_INPUT_DATA_CONFIG_FILE), "r") as f:
        data_config = json.load(f)

    checkpoint_config_file = os.getenv(sm_env_constants.SM_CHECKPOINT_CONFIG_FILE)
    if os.path.exists(checkpoint_config_file):
        with open(checkpoint_config_file, "r") as f:
            checkpoint_config = json.load(f)
    else:
        checkpoint_config = {}

    train_path = os.environ[sm_env_constants.SM_CHANNEL_TRAIN]
    val_path = os.environ.get(sm_env_constants.SM_CHANNEL_VALIDATION)

    if train_path == val_path or os.path.basename(train_path) == os.path.basename(val_path):
        logger.warning('Found same path for training and validation. This is not recommended and results may not '
                       'be correct.')
    else:
        # Check if there is potential data redundancy between training and validation sets
        check_data_redundancy(train_path, val_path)

    sm_hosts = json.loads(os.environ[sm_env_constants.SM_HOSTS])
    sm_current_host = os.environ[sm_env_constants.SM_CURRENT_HOST]

    model_dir = os.getenv(sm_env_constants.SM_MODEL_DIR)

    sagemaker_train(
        train_config=train_config, data_config=data_config,
        train_path=train_path, val_path=val_path, model_dir=model_dir,
        sm_hosts=sm_hosts, sm_current_host=sm_current_host,
        checkpoint_config=checkpoint_config
        )


def train(training_environment):
    """Run XGBoost training in either algorithm mode or using a user supplied module in local SageMaker environment.
    The user supplied module and its dependencies are downloaded from S3.
    Training is invoked by calling a "train" function in the user supplied module.

    Args:
        training_environment: training environment object containing environment variables,
                               training arguments and hyperparameters
    """
    if training_environment.user_entry_point is not None:
        logger.info('Invoking user training script.')
        framework.modules.run_module(training_environment.module_dir, training_environment.to_cmd_args(),
                                     training_environment.to_env_vars(), training_environment.module_name,
                                     capture_error=False)
    else:
        logger.info("Running XGBoost Sagemaker in algorithm mode")
        _env.write_env_vars(training_environment.to_env_vars())

        run_algorithm_mode()


def main():
    train(framework.training_env())
    sys.exit(0)
