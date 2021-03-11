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

from sagemaker_containers.beta.framework import (
    encoders,
    env,
    modules,
    server,
    transformer,
    worker,
)

from sagemaker_algorithm_toolkit import exceptions as exc
from sagemaker_xgboost_container import encoder as xgb_encoders
from sagemaker_xgboost_container.algorithm_mode import serve
from sagemaker_xgboost_container.serving_mms import start_mxnet_model_server

logging.basicConfig(
    format="%(asctime)s %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logging.getLogger("boto3").setLevel(logging.INFO)
logging.getLogger("s3transfer").setLevel(logging.INFO)
logging.getLogger("botocore").setLevel(logging.WARN)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def is_multi_model():
    return os.environ.get("SAGEMAKER_MULTI_MODEL")


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
        raise exc.UserError(
            "Cannot use transform_fn implementation with input_fn, predict_fn, and/or output_fn"
        )

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
            user_module = modules.import_module(
                serving_env.module_dir, serving_env.module_name
            )
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
    max_content_length = os.getenv("MAX_CONTENT_LENGTH", DEFAULT_MAX_CONTENT_LEN)
    if int(max_content_length) > MAX_CONTENT_LEN_LIMIT:
        # Cap at 20mb
        max_content_length = MAX_CONTENT_LEN_LIMIT

    max_workers = multiprocessing.cpu_count()
    max_job_queue_size = 2*max_workers
    
    logging.info("Max Workers: {}".format(max_workers))
    logging.info("Max Job Queue Size: {}".format(max_job_queue_size))
    
    # Max heap size = (max workers + max job queue size) * max payload size * 1.2 (20% buffer) + 128 (base amount)
    max_heap_size = ceil((max_workers + max_job_queue_size) * (int(max_content_length) / 1024**2) * 1.2) + 128

    if is_multi_model:
 #       os.environ["SAGEMAKER_NUM_MODEL_WORKERS"] = '16'
        os.environ["SAGEMAKER_NUM_MODEL_WORKERS"] = str(max_workers*2)
 #       os.environ["SAGEMAKER_MODEL_JOB_QUEUE_SIZE"] = '400'
        os.environ["SAGEMAKER_MODEL_JOB_QUEUE_SIZE"] = str(max_workers*2)
        os.environ["SAGEMAKER_MMS_MODEL_STORE"] = '/'
        os.environ["SAGEMAKER_MMS_LOAD_MODELS"] = ''
    else:
        os.environ["SAGEMAKER_NUM_MODEL_WORKERS"] = str(max_workers)
        os.environ["SAGEMAKER_MODEL_JOB_QUEUE_SIZE"] = str(max_job_queue_size)
        os.environ["SAGEMAKER_MMS_MODEL_STORE"] = '/opt/ml/model'
        os.environ["SAGEMAKER_MMS_LOAD_MODELS"] = 'ALL'

    if not os.getenv("SAGEMAKER_BIND_TO_PORT", None):
        os.environ["SAGEMAKER_BIND_TO_PORT"] = str(PORT)

    os.environ["SAGEMAKER_MAX_HEAP_SIZE"] = str(max_heap_size) + 'm'
    os.environ["SAGEMAKER_MAX_DIRECT_MEMORY_SIZE"] = os.environ["SAGEMAKER_MAX_HEAP_SIZE"]

    os.environ["SAGEMAKER_MAX_REQUEST_SIZE"] = str(max_content_length)
    os.environ["SAGEMAKER_MMS_DEFAULT_HANDLER"] = handler

    # TODO: Revert config.properties.tmp to config.properties and add back in vmargs
    # set with environment variables after MMS implements parsing environment variables
    # for vmargs, update MMS section of final/Dockerfile.cpu to match, and remove the
    # following code.
    try:
        with open('/home/model-server/config.properties.tmp', 'r') as f:
            with open('/home/model-server/config.properties', 'w+') as g:
                g.write("vmargs=-XX:-UseLargePages -XX:-UseContainerSupport -XX:+UseG1GC -XX:MaxMetaspaceSize=32M -XX:+ExitOnOutOfMemoryError "
                        + "-Xmx" + os.environ["SAGEMAKER_MAX_HEAP_SIZE"]
                        + " -XX:MaxDirectMemorySize=" + os.environ["SAGEMAKER_MAX_DIRECT_MEMORY_SIZE"] + "\n")
                g.write(f.read())
    except Exception:
        pass


def main():
    serving_env = env.ServingEnv()
    is_multi_model = _is_multi_model_endpoint()

    if serving_env.module_name is None:
        logging.info("Starting MXNet server in algorithm mode.")
        _start_model_server(is_multi_model, ALGO_HANDLER_SERVICE)
    else:
        server.start(env.ServingEnv().framework_module)
