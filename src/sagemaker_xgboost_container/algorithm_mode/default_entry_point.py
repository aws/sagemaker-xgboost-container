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
import sys

from sagemaker_containers import entry_point
from sagemaker_algorithm_toolkit import exceptions

from sagemaker_xgboost_container import distributed
from sagemaker_xgboost_container.algorithm_mode import channel_validation as cv
from sagemaker_xgboost_container.algorithm_mode import hyperparameter_validation as hpv
from sagemaker_xgboost_container.algorithm_mode import metrics as metrics_mod
from sagemaker_xgboost_container.algorithm_mode.train_helper import train_job
from sagemaker_xgboost_container.constants import sm_env_constants


logger = logging.getLogger(__name__)


def _has_train_data(train_path=None):
    train_path = train_path if train_path else os.environ[sm_env_constants.SM_CHANNEL_TRAIN]
    logger.info("Reading Train Data from {}".format(train_path))
    train_files = [os.path.join(train_path, train_file) for train_file in os.listdir(train_path)]

    return len(train_files) > 0


def algorithm_mode_train():
    """Train XGBoost single node or distributed mode."""
    # TODO: replace with CSDK constants in sagemaker_containers._env
    INPUT_TRAIN_CONFIG_PATH = os.getenv("SM_INPUT_TRAINING_CONFIG_FILE")
    INPUT_DATA_CONFIG_PATH = os.getenv("SM_INPUT_DATA_CONFIG_FILE")
    train_config = json.load(open(INPUT_TRAIN_CONFIG_PATH, "r"))
    data_config = json.load(open(INPUT_DATA_CONFIG_PATH, "r"))

    metrics = metrics_mod.initialize()

    hyperparameters = hpv.initialize(metrics)
    validated_train_config = hyperparameters.validate(train_config)
    if validated_train_config.get("updater"):
        validated_train_config["updater"] = ",".join(validated_train_config["updater"])

    channels = cv.initialize()
    validated_data_config = channels.validate(data_config)

    logging.debug("hyperparameters {}".format(validated_train_config))
    logging.debug("channels {}".format(validated_data_config))

    # Obtain information about training resources to determine whether to set up Rabit or not
    sm_hosts = json.loads(os.environ[sm_env_constants.SM_HOSTS])
    sm_current_host = os.environ[sm_env_constants.SM_CURRENT_HOST]
    num_hosts = len(sm_hosts)

    if num_hosts > 1:
        # Wait for hosts to find each other
        logging.info("Distributed node training with {} hosts: {}".format(num_hosts, sm_hosts))
        entry_point._wait_hostname_resolution()

        train_with_rabit(validated_train_config, validated_data_config, sm_hosts, sm_current_host)
    elif num_hosts == 1:
        logging.info("Single node training.")
        train_job(validated_train_config, validated_data_config, False, True)
    else:
        raise exceptions.PlatformError("Number of hosts should be an int greater than or equal to 1")


def train_with_rabit(train_config, data_config, sm_hosts, sm_current_host):
    """Initialize Rabit Tracker and run training.

    Rabit is initialized twice.
    The first initialization is to make sure the hosts can find each other, then broadcast whether
    or not training data exists on the current host. If there is only 1 host with training data, this
    becomes a single node training job.

    Otherwise, the hosts with training data are used for the second Rabit set up. In algorithm mode,
    the is_master boolean flag is used to determine whether the current host is the Rabit master.

    :param train_config:
    :param data_config:
    :param sm_hosts:
    :param sm_current_host:
    """
    has_data = _has_train_data()

    with distributed.Rabit(hosts=sm_hosts, current_host=sm_current_host) as rabit:
        hosts_with_data = rabit.synchronize({'host': rabit.current_host, 'has_data': has_data})
        hosts_with_data = [record['host'] for record in hosts_with_data if record['has_data']]

        # Keep track of port used, so that hosts trying to shutdown know when server is not available
        previous_port = rabit.master_port

    if not has_data:
        logger.warning("No data was available for training on current host. Exiting from training job.")
        sys.exit(0)

    if len(hosts_with_data) > 1:
        # Set up Rabit with nodes that have data and an unused port so that previous slaves don't confuse it
        # with the previous rabit configuration
        with distributed.Rabit(hosts=hosts_with_data,
                               current_host=sm_current_host,
                               port=previous_port + 1) as cluster:
            train_job(train_config, data_config, is_distributed=True, is_master=cluster.is_master)

    elif len(hosts_with_data) == 1:
        logging.debug("Only 1 host with training data, "
                      "starting single node training job from: {}".format(sm_current_host))
        train_job(train_config, data_config, is_distributed=False, is_master=True)

    else:
        raise exceptions.PlatformError("No hosts received training data.")