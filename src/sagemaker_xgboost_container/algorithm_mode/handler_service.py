# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import os

from sagemaker_inference import content_types, default_inference_handler, encoder
from sagemaker_inference.default_handler_service import DefaultHandlerService

from sagemaker_xgboost_container.algorithm_mode import serve_utils
from sagemaker_xgboost_container.algorithm_mode.inference_errors import (
    BadRequestInferenceError,
    ModelLoadInferenceError,
    NoContentInferenceError,
    UnsupportedMediaTypeInferenceError,
)
from sagemaker_xgboost_container.mms_patch.mms_transformer import XGBMMSTransformer

SAGEMAKER_BATCH = os.getenv("SAGEMAKER_BATCH")


class HandlerService(DefaultHandlerService):
    """Handler service that is executed by the model server.
    Determines specific default inference handlers to use based on the type MXNet model being used.
    This class extends ``DefaultHandlerService``, which define the following:
        - The ``handle`` method is invoked for all incoming inference requests to the model server.
        - The ``initialize`` method is invoked at model server start up.
    Based on: https://github.com/awslabs/mxnet-model-server/blob/v1.0.8/docs/custom_service.md
    """

    class DefaultXGBoostAlgoModeInferenceHandler(default_inference_handler.DefaultInferenceHandler):
        def default_model_fn(self, model_dir):
            """Load a model. For XGBoost Framework, a default function to load a model is not provided.
            Users should provide customized model_fn() in script.
            Args:
                model_dir: a directory where model is saved.
            Returns:
                A XGBoost model.
                XGBoost model format type.
            """
            try:
                booster, format = serve_utils.get_loaded_booster(model_dir, serve_utils.is_ensemble_enabled())
            except Exception as e:
                raise ModelLoadInferenceError("Unable to load model: {}".format(str(e)))
            return booster, format

        def default_input_fn(self, input_data, input_content_type):
            """Take request data and de-serializes the data into an object for prediction.
                When an InvokeEndpoint operation is made against an Endpoint running SageMaker model server,
                the model server receives two pieces of information:
                    - The request Content-Type, for example "application/json"
                    - The request data, which is at most 5 MB (5 * 1024 * 1024 bytes) in size.
                The input_fn is responsible to take the request data and pre-process it before prediction.
            Args:
                input_data (obj): the request data.
                input_content_type (str): the request Content-Type. XGBoost accepts CSV, LIBSVM, and RECORDIO-PROTOBUF.
            Returns:
                (obj): data ready for prediction. For XGBoost, this defaults to DMatrix.
            """
            if len(input_data) == 0:
                raise NoContentInferenceError()
            dtest, content_type = serve_utils.parse_content_data(input_data, input_content_type)
            return dtest, content_type

        def default_predict_fn(self, data, model):
            """A default predict_fn for XGBooost Framework. Calls a model on data deserialized in input_fn.
            Args:
                data: input data (DMatrix) for prediction deserialized by input_fn and data content type
                model: XGBoost model loaded in memory by model_fn, and xgboost model format
            Returns: a prediction
            """
            booster, model_format = model
            dtest, content_type = data
            try:
                return serve_utils.predict(booster, model_format, dtest, content_type)
            except Exception as e:
                raise BadRequestInferenceError(str(e))

        def default_output_fn(self, prediction, accept):
            """Return encoded prediction for the response.
            Args:
                prediction (obj): prediction returned by predict_fn .
                accept (str): accept content-type expected by the client.
            Returns:
                encoded response for MMS to return to client
            """
            accept_type = accept.lower()
            try:
                if accept_type == content_types.CSV or accept_type == "csv":
                    if SAGEMAKER_BATCH:
                        return_data = "\n".join(map(str, prediction.tolist())) + "\n"
                    else:
                        # FIXME: this is invalid CSV and is only retained for backwards compatibility
                        return_data = ",".join(map(str, prediction.tolist()))
                    encoded_prediction = return_data.encode("utf-8")
                elif accept_type == content_types.JSON or accept_type == "json":
                    encoded_prediction = encoder.encode(prediction, accept_type)
                else:
                    raise ValueError(
                        "{} is not an accepted Accept type. Please choose one of the following:"
                        " ['{}', '{}'].".format(accept, content_types.CSV, content_types.JSON)
                    )
            except Exception as e:
                raise UnsupportedMediaTypeInferenceError(
                    "Encoding to accept type {} failed with exception: {}".format(accept, e)
                )
            return encoded_prediction

    def __init__(self):
        transformer = XGBMMSTransformer(default_inference_handler=self.DefaultXGBoostAlgoModeInferenceHandler())
        super(HandlerService, self).__init__(transformer=transformer)
