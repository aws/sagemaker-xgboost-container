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
# import os
# import pkg_resources
import subprocess
# import sys
# import time

# from retrying import retry

# from sagemaker_containers.beta.framework import (encoders, env, modules, transformer, worker)

# from sagemaker_xgboost_container import encoder as xgb_encoders

# import sagemaker_inference
from sagemaker_inference import model_server
# from sagemaker_inference.model_server import _create_model_server_config_file, _add_sigterm_handler
# from sagemaker_inference import default_handler_service, environment, utils
from sagemaker_xgboost_container.algorithm_mode import handler_service


HANDLER_SERVICE = handler_service.__name__


def _retry_if_error(exception):
    return isinstance(exception, subprocess.CalledProcessError)


# @retry(stop_max_delay=1000 * 30, retry_on_exception=_retry_if_error)
def _start_model_server():
    # there's a race condition that causes the model server command to
    # sometimes fail with 'bad address'. more investigation needed
    # retry starting mms until it's ready
    print("Trying to set up model server handler + {}".format(HANDLER_SERVICE))
    model_server.start_model_server(handler_service=HANDLER_SERVICE)


def main():  # environ, start_response):
    # serving_env = env.ServingEnv()
    # if serving_env.module_name is None:
    print("Starting MXNet server")
    _start_model_server()
