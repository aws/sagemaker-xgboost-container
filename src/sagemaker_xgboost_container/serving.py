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
import multiprocessing
import os
import subprocess

from retrying import retry
from sagemaker_containers.beta.framework import env, modules

from sagemaker_xgboost_container.mms_patch import model_server
from sagemaker_xgboost_container.algorithm_mode import handler_service as algo_handler_service
from sagemaker_xgboost_container import handler_service as user_module_handler_service


ALGO_HANDLER_SERVICE = algo_handler_service.__name__
USER_HANDLER_SERVICE = user_module_handler_service.__name__

PORT = 8080
DEFAULT_MAX_CONTENT_LEN = 6 * 1024**2


def _retry_if_error(exception):
    return isinstance(exception, subprocess.CalledProcessError)


@retry(stop_max_delay=1000 * 30, retry_on_exception=_retry_if_error)
def _start_model_server(is_multi_model, handler):
    # there's a race condition that causes the model server command to
    # sometimes fail with 'bad address'. more investigation needed
    # retry starting mms until it's ready
    logging.info("Trying to set up model server handler: {}".format(handler))
    _set_mms_configs(is_multi_model, handler)
    model_server.start_model_server(handler_service=handler,
                                    is_multi_model=is_multi_model,
                                    config_file=os.environ['XGBOOST_MMS_CONFIG'])


def _is_multi_model_endpoint():
    if "SAGEMAKER_MULTI_MODEL" in os.environ and os.environ["SAGEMAKER_MULTI_MODEL"] == 'true':
        return True
    else:
        return False


def _set_mms_configs(is_multi_model, handler):
    """Set environment variables for MMS to parse during server initialization. These env vars are used to
    propagate the config.properties file used during MxNet Model Server initialization.

    'SAGEMAKER_MMS_MODEL_STORE' has to be set to the model location during single model inference because MMS
    is initialized with the model. In multi-model mode, MMS is started with no models loaded.

    Note: Ideally, instead of relying on env vars, this should be written directly to a config file.
    """
    max_content_length = os.getenv("MAX_CONTENT_LENGTH", DEFAULT_MAX_CONTENT_LEN)

    if is_multi_model:
        os.environ["SAGEMAKER_NUM_MODEL_WORKERS"] = '1'
        os.environ["SAGEMAKER_MMS_MODEL_STORE"] = '/'
        os.environ["SAGEMAKER_MMS_LOAD_MODELS"] = ''
    else:
        os.environ["SAGEMAKER_NUM_MODEL_WORKERS"] = str(multiprocessing.cpu_count())
        os.environ["SAGEMAKER_MMS_MODEL_STORE"] = '/opt/ml/model'
        os.environ["SAGEMAKER_MMS_LOAD_MODELS"] = 'ALL'

    if not os.getenv("SAGEMAKER_BIND_TO_PORT", None):
        os.environ["SAGEMAKER_BIND_TO_PORT"] = str(PORT)

    os.environ["SAGEMAKER_MAX_REQUEST_SIZE"] = str(max_content_length)
    os.environ["SAGEMAKER_MMS_DEFAULT_HANDLER"] = handler


def main():
    serving_env = env.ServingEnv()
    is_multi_model = _is_multi_model_endpoint()

    if serving_env.module_name is None:
        logging.info("Starting MXNet server in algorithm mode.")
        _start_model_server(is_multi_model, ALGO_HANDLER_SERVICE)
    else:
        logging.info("Staring MXNet Model Server with user module.")
        # Install user module from s3 to import
        modules.import_module(serving_env.module_dir, serving_env.module_name)
        _start_model_server(is_multi_model, USER_HANDLER_SERVICE)
