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

from sagemaker_xgboost_container.algorithm_mode import channel_validation as cv
from sagemaker_xgboost_container.algorithm_mode import hyperparameter_validation as hpv
from sagemaker_xgboost_container.algorithm_mode import metrics as metrics_mod
from sagemaker_xgboost_container.algorithm_mode.train_helper import train_job, get_size, \
    validate_file_format, EASE_MEMORY


INPUT_DATA_PATH = os.getenv("ALGO_INPUT_DATA_DIR")
HTTP_SERVER_PORT = "8000"


def algorithm_mode_train(is_master):
    """Train XGBoost single node or distributed mode.

    Note that in algorithm mode, the is_master boolean is used to determine which node stores the trained model to S3.
    In script mode the booleana can be computed by comparing the current host name with the host name
    stored in the env var 'SM_MASTER'.

    :param is_master: True if single node training, or if the current host is the master node described in the env
                      var 'SM_MASTER'. The master host should be the host to save the trained model to S3.
    """
    # TODO: replace with CSDK constants in sagemaker_containers._env
    INPUT_TRAIN_CONFIG_PATH = os.getenv("SM_INPUT_TRAINING_CONFIG_FILE")
    INPUT_DATA_CONFIG_PATH = os.getenv("SM_INPUT_DATA_CONFIG_FILE")
    INPUT_RESOURCE_CONFIG_PATH = os.getenv("SM_INPUT_RESOURCE_CONFIG_FILE")

    resource_config = json.load(open(INPUT_RESOURCE_CONFIG_PATH, "r"))
    train_config = json.load(open(INPUT_TRAIN_CONFIG_PATH, "r"))
    data_config = json.load(open(INPUT_DATA_CONFIG_PATH, "r"))

    metrics = metrics_mod.initialize()

    hyperparameters = hpv.initialize(metrics)
    final_train_config = hyperparameters.validate(train_config)

    channels = cv.initialize()
    final_data_config = channels.validate(data_config)

    logging.info("hyperparameters {}".format(final_train_config))
    logging.info("channels {}".format(final_data_config))

    train_job(resource_config, final_train_config, final_data_config, is_master)
