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
import json
import logging
import os

from sagemaker_algorithm_toolkit import exceptions
from sagemaker_xgboost_container.algorithm_mode import channel_validation as cv
from sagemaker_xgboost_container.algorithm_mode import hyperparameter_validation as hpv
from sagemaker_xgboost_container.algorithm_mode import metrics as metrics_mod
from sagemaker_xgboost_container.algorithm_mode import train_helper


def algorithm_mode_train():
    # TODO: replace with CSDK constants in sagemaker_containers._env
    INPUT_TRAIN_CONFIG_PATH = os.getenv("ALGO_INPUT_TRAINING_CONFIG_FILE")
    INPUT_DATA_CONFIG_PATH = os.getenv("ALGO_INPUT_DATA_CONFIG_FILE")
    INPUT_RESOURCE_CONFIG_PATH = os.getenv("ALGO_INPUT_RESOURCE_CONFIG_FILE")
    # END TODO

    resource_config = json.load(open(INPUT_RESOURCE_CONFIG_PATH, "r"))
    train_config = json.load(open(INPUT_TRAIN_CONFIG_PATH, "r"))
    data_config = json.load(open(INPUT_DATA_CONFIG_PATH, "r"))

    # TODO: implement distributed training
    num_hosts = len(resource_config["hosts"])
    if num_hosts < 1:
        raise exceptions.PlatformError("Number of hosts should be greater or equal to 1")
    elif num_hosts > 1:
        raise exceptions.UserError("Running distributed xgboost training; this is not supported yet.")
    # END TODO

    metrics = metrics_mod.initialize()

    hyperparameters = hpv.initialize(metrics)
    final_train_config = hyperparameters.validate(train_config)

    channels = cv.initialize()
    final_data_config = channels.validate(data_config)

    logging.info("hyperparameters {}".format(final_train_config))
    logging.info("channels {}".format(final_data_config))

    train_helper.train_job(resource_config, final_train_config, final_data_config)
