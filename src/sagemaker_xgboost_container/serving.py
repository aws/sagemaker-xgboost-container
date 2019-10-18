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
import subprocess

from retrying import retry

from sagemaker_containers.beta.framework import env

from sagemaker_inference import model_server
from sagemaker_xgboost_container.algorithm_mode import handler_service as algo_handler_service
from sagemaker_xgboost_container import handler_service as user_module_handler_service


ALGO_HANDLER_SERVICE = algo_handler_service.__name__
USER_HANDLER_SERVICE = user_module_handler_service.__name__


def _retry_if_error(exception):
    return isinstance(exception, subprocess.CalledProcessError)


@retry(stop_max_delay=1000 * 30, retry_on_exception=_retry_if_error)
def _start_model_server(handler):
    # there's a race condition that causes the model server command to
    # sometimes fail with 'bad address'. more investigation needed
    # retry starting mms until it's ready
    logging.info("Trying to set up model server handler: {}".format(handler))
    model_server.start_model_server(handler_service=handler)


def main():
    serving_env = env.ServingEnv()

    if serving_env.module_name is None:
        logging.info("Starting MXNet server in algorithm mode.")
        _start_model_server(ALGO_HANDLER_SERVICE)
    else:
        logging.info("Staring MXNet Model Server with user module.")
        _start_model_server(USER_HANDLER_SERVICE)
