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

from sagemaker_containers import _env, entry_point
import sagemaker_containers.beta.framework as framework
from sagemaker_algorithm_toolkit import exceptions
from sagemaker_xgboost_container import distributed
from sagemaker_xgboost_container.algorithm_mode import default_entry_point


logger = logging.getLogger(__name__)


HOSTS = 'SM_HOSTS'
MASTER_HOST = 'SM_RABIT_MASTER'
CURRENT_HOST = 'current_host'


def _has_train_data(train_path=None):
    train_path = train_path if train_path else os.environ['SM_CHANNEL_TRAIN']
    logger.info("Reading Train Data from {}".format(train_path))
    train_files = [os.path.join(train_path, train_file) for train_file in os.listdir(train_path)]

    return len(train_files) > 0


def train(training_environment, is_master=True):
    """ Runs XGBoost training on a user supplied module in local SageMaker environment.
    The user supplied module and its dependencies are downloaded from S3.
    Training is invoked by calling a "train" function in the user supplied module.

    Args:
        training_environment: training environment object containing environment variables,
                               training arguments and hyperparameters
        is_master: True if training using a single node, or master host in distributed training.
                   To use in algorithm mode; in script mode, user entry point must parse environment var
                   'SM_RABIT_MASTER'
    """
    if training_environment.user_entry_point is not None:
        logger.info('Invoking user training script.')
        framework.modules.run_module(training_environment.module_dir, training_environment.to_cmd_args(),
                                     training_environment.to_env_vars(), training_environment.module_name)
    else:
        logger.info("Running XGBoost Sagemaker in algorithm mode")
        # Write environment variables for algorithm mode
        logging.info(training_environment.to_env_vars())
        _env.write_env_vars(training_environment.to_env_vars())
        default_entry_point.algorithm_mode_train(is_master)


def setup_rabit_and_train(training_env, sm_hosts, sm_current_host):
    """Initialize Rabit Tracker and run training.

    Rabit is initialized twice.
    The first initialization is to make sure the hosts can find each other, then broadcast whether
    or not training data exists on the current host. If there is only 1 host with training data, this
    becomes a single node training job.

    Otherwise, only the hosts with training data are used for the second Rabit set up. In algorithm mode,
    the is_master boolean flag is used to determine whether the current host is the Rabit master.
    In script mode, the script can parse this information from the 'SM_IS_RABIT_MASTER', which is set to
    'True' only on the master host.


    :param training_env:
    :param sm_hosts:
    :param sm_current_host:
    """
    has_data = _has_train_data()

    with distributed.Rabit(hosts=sm_hosts, current_host=sm_current_host) as rabit:
        hosts_with_data = rabit.synchronize({'host': rabit.current_host, 'has_data': has_data})
        hosts_with_data = [record['host'] for record in hosts_with_data if record['has_data']]

        # Keep track of port used, so that hosts trying to shutdown know when server is not available
        # TODO: Remove when rabit.finalize synchronizes the slave and master shutdown
        previous_port = rabit.master_port

    if not has_data:
        logger.warning("No data was available for training on current host. Exiting.")
        sys.exit(0)

    if len(hosts_with_data) > 1:
        # Set up Rabit with nodes that have data and an unused port so that previous slaves don't confuse it
        # with the previous rabit configuration
        with distributed.Rabit(hosts=hosts_with_data,
                               current_host=sm_current_host,
                               port=previous_port + 1) as cluster:
            os.environ['SM_HOSTS'] = json.dumps(hosts_with_data)
            if cluster.is_master:
                os.environ['SM_IS_RABIT_MASTER'] = 'True'  # Set master host for access during script mode
            else:
                os.environ['SM_IS_RABIT_MASTER'] = 'False'

            train(training_env, cluster.is_master)

    elif len(hosts_with_data) == 1:
        logging.debug("Only 1 host with training data, "
                      "starting single node training job from: {}".format(sm_current_host))
        train(training_env, True)

    else:
        raise exceptions.PlatformError("No hosts received training data.")


def main():
    training_env = framework.training_env()

    # Obtain information about training resources to determine whether to set up Rabit or not
    sm_hosts = training_env.hosts
    sm_current_host = training_env.resource_config['current_host']
    num_hosts = len(sm_hosts)

    if num_hosts > 1:
        # Wait for hosts to find each other
        logging.info("Distributed node training!")
        entry_point._wait_hostname_resolution()

        setup_rabit_and_train(training_env, sm_hosts, sm_current_host)
    elif num_hosts == 1:
        logging.info("Single node training!")
        train(training_env, True)
    else:
        raise exceptions.PlatformError("Number of hosts should be greater or equal to 1")

    sys.exit(0)
