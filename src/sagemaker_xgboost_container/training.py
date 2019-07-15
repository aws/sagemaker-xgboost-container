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

import logging
import sys

from sagemaker_containers import _env
import sagemaker_containers.beta.framework as framework
from sagemaker_xgboost_container.algorithm_mode import default_entry_point

logger = logging.getLogger(__name__)


def train(training_environment):
    """Runs XGBoost training in either algorithm mode or using a user supplied module in local SageMaker environment.
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
                                     capture_error=True)
    else:
        logger.info("Running XGBoost Sagemaker in algorithm mode")
        _env.write_env_vars(training_environment.to_env_vars())
        default_entry_point.algorithm_mode_train()


def main():
    train(framework.training_env())
    sys.exit(0)
