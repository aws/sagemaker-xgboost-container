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
import os
import importlib

from sagemaker_containers.beta.framework import (
    encoders,
    env,
    transformer,
    worker,
)

from sagemaker_algorithm_toolkit import exceptions as exc
from sagemaker_xgboost_container import encoder as xgb_encoders
from sagemaker_xgboost_container.algorithm_mode import serve
from sagemaker_xgboost_container.constants import sm_env_constants
from sagemaker_xgboost_container.sagemaker_containers_patch import server
from sagemaker_xgboost_container.serving_mms import start_mxnet_model_server

logging.basicConfig(format="%(asctime)s %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logging.getLogger("boto3").setLevel(logging.INFO)
logging.getLogger("s3transfer").setLevel(logging.INFO)
logging.getLogger("botocore").setLevel(logging.WARN)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def is_multi_model():
    return os.environ.get("SAGEMAKER_MULTI_MODEL")


def set_default_serving_env_if_unspecified():
    """Set default values for environment variables if they aren't already specified.

    set "OMP_NUM_THREADS" = sm_env_constants.ONE_THREAD_PER_PROCESS
    Single-thread processes by default. Multithreading can introduce significant
    performance overhead due to task switching.
    """
    env_default_dict = {"OMP_NUM_THREADS": sm_env_constants.ONE_THREAD_PER_PROCESS}
    for always_specified_key, default_value in env_default_dict.items():
        try:
            # If this does not throw, the user has specified a non-default value.
            os.environ[always_specified_key]
        except KeyError:
            #  Key that is always specified is not set in the environment. Set default value.
            os.environ[always_specified_key] = default_value


def default_model_fn(model_dir):
    """Load a model. For XGBoost Framework, a default function to load a model is not provided.
    Users should provide customized model_fn() in script.
    Args:
        model_dir: a directory where model is saved.
    Returns: A XGBoost model.
    """
    return transformer.default_model_fn(model_dir)


def default_input_fn(input_data, content_type):
    """Take request data and de-serializes the data into an object for prediction.
        When an InvokeEndpoint operation is made against an Endpoint running SageMaker model server,
        the model server receives two pieces of information:
            - The request Content-Type, for example "application/json"
            - The request data, which is at most 5 MB (5 * 1024 * 1024 bytes) in size.
        The input_fn is responsible to take the request data and pre-process it before prediction.
        Note: For CSV data, the decoder will error if there are any leading or trailing newline
        chars.
    Args:
        input_data (obj): the request data.
        content_type (str): the request Content-Type.
    Returns:
        (obj): data ready for prediction. For XGBoost, this defaults to DMatrix.
    """
    return xgb_encoders.decode(input_data, content_type)


def default_predict_fn(input_data, model):
    """A default predict_fn for XGBooost Framework. Calls a model on data deserialized in input_fn.
    Args:
        input_data: input data (Numpy array) for prediction deserialized by input_fn
        model: XGBoost model loaded in memory by model_fn
    Returns: a prediction
    """
    output = model.predict(input_data, validate_features=False)
    return output


def default_output_fn(prediction, accept):
    """Function responsible to serialize the prediction for the response.
    Args:
        prediction (obj): prediction returned by predict_fn .
        accept (str): accept content-type expected by the client.
    Returns:
        (worker.Response): a Flask response object with the following args:
            * Args:
                response: the serialized data to return
                accept: the content-type that the data was transformed to.
    """
    return worker.Response(encoders.encode(prediction, accept), mimetype=accept)


def _user_module_transformer(user_module):
    model_fn = getattr(user_module, "model_fn", default_model_fn)
    input_fn = getattr(user_module, "input_fn", None)
    predict_fn = getattr(user_module, "predict_fn", None)
    output_fn = getattr(user_module, "output_fn", None)
    transform_fn = getattr(user_module, "transform_fn", None)

    if transform_fn and (input_fn or predict_fn or output_fn):
        raise exc.UserError("Cannot use transform_fn implementation with input_fn, predict_fn, and/or output_fn")

    if transform_fn is not None:
        return transformer.Transformer(model_fn=model_fn, transform_fn=transform_fn)
    else:
        return transformer.Transformer(
            model_fn=model_fn,
            input_fn=input_fn or default_input_fn,
            predict_fn=predict_fn or default_predict_fn,
            output_fn=output_fn or default_output_fn,
        )


app = None


def main(environ, start_response):
    global app
    if app is None:
        serving_env = env.ServingEnv()
        if serving_env.module_name is None:
            app = serve.ScoringService.csdk_start()
        else:
            user_module = importlib.import_module(serving_env.module_name)
            user_module_transformer = _user_module_transformer(user_module)
            user_module_transformer.initialize()
            app = worker.Worker(
                transform_fn=user_module_transformer.transform,
                module_name=serving_env.module_name,
            )

    return app(environ, start_response)


def serving_entrypoint():
    """Start Inference Server.

    NOTE: If the inference server is multi-model, MxNet Model Server will be used as the base server. Otherwise,
        GUnicorn is used as the base server.
    """
    set_default_serving_env_if_unspecified()

    if is_multi_model():
        start_mxnet_model_server()
    else:
        server.start(env.ServingEnv().framework_module)
