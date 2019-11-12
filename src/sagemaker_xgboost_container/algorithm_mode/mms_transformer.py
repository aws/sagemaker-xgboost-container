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

import importlib
import logging

from sagemaker_inference import content_types, environment, utils
from sagemaker_inference.default_inference_handler import DefaultInferenceHandler

from sagemaker_xgboost_container.algorithm_mode.inference_errors import BaseInferenceToolkitError
"""
This is a modified copy of the Transformer found in sagemaker-inference, which is found here:
https://github.com/aws/sagemaker-inference-toolkit/blob/master/src/sagemaker_inference/transformer.py

This Transformer was written for the addition of:
1. Error handling that responds back to client, rather than failing with an Intenal Server Error
2. Not failing when there is no user module, which is needed to support XGBoost algorithm mode

Ideally, these changes, as well as the InferenceToolkitError, should be moved to the SageMaker inference toolkit.
"""


class Transformer(object):
    """Represents the execution workflow for handling inference requests sent to the model server."""
    def __init__(self, default_inference_handler=None):
        """Initialize a ``Transformer``.
        Args:
            default_inference_handler (DefaultInferenceHandler): default implementation of inference handlers to
                use in absence of expected serving functions within the user module.
                Defaults to ``DefaultInferenceHandler``.
        """
        self._default_inference_handler = default_inference_handler or DefaultInferenceHandler()
        self._initialized = False
        self._environment = None
        self._model = None

        self._model_fn = None
        self._transform_fn = None
        self._input_fn = None
        self._predict_fn = None
        self._output_fn = None

    @staticmethod
    def handle_error(context, status_code, message):
        # This model is created to test reporting of an error in a batch of requests
        context.set_response_status(code=status_code,
                                    phrase=message,
                                    idx=0)
        return [message]

    def transform(self, data, context):
        """Take a request with input data, deserialize it, make a prediction, and return a
        serialized response.
        Args:
            data (obj): the request data.
            context (obj): metadata on the incoming request data.
        Returns:
            list[obj]: the serialized prediction result wrapped in a list.
        """
        if not self._initialized:
            try:
                sys_properties = context._system_properties
                model_dir = sys_properties.get('model_dir')

                self.validate_and_initialize(model_dir)
            except Exception as e:
                if isinstance(e, BaseInferenceToolkitError):
                    logging.error("Error loading model: {}".format(e))
                    return self.handle_error(context, e.status_code, e.message)
                else:
                    raise e
            self._initialized = True

        try:
            input_data = data[0].get('body')

            request_processor = context.request_processor[0]

            request_property = request_processor.get_request_properties()
            content_type = utils.retrieve_content_type_header(request_property)
            accept = request_property.get('Accept') or request_property.get('accept')

            if not accept or accept == content_types.ANY:
                accept = self._environment.default_accept

            result = self._transform_fn(self._model, input_data, content_type, accept)

            response = result
            response_content_type = accept

            if isinstance(result, tuple):
                # handles tuple for backwards compatibility
                response = result[0]
                response_content_type = result[1]

            context.set_response_content_type(0, response_content_type)
            return [response]
        except Exception as e:
            if isinstance(e, BaseInferenceToolkitError):
                logging.error(e)
                return self.handle_error(context, e.status_code, e.message)
            else:
                raise e

    def validate_and_initialize(self, model_dir):  # type: () -> None
        """Validates the user module against the SageMaker inference contract.
        Load the model as defined by the ``model_fn`` to prepare handling predictions.

        NOTE: This still uses environment values from the legacy sagemaker-containers. This should be removed.
        """
        self._environment = environment.Environment()
        self._validate_user_module_and_set_functions()
        self._model = self._model_fn(model_dir)
        self._initialized = True

    def _validate_user_module_and_set_functions(self):
        """Retrieves and validates the inference handlers provided within the user module.
        Default implementations of the inference handlers are utilized in place of missing functions defined
        in the user module.
        """
        user_module_name = self._environment.module_name
        try:
            user_module = importlib.import_module(user_module_name)

            self._model_fn = getattr(user_module, 'model_fn', self._default_inference_handler.default_model_fn)

            transform_fn = getattr(user_module, 'transform_fn', None)
            input_fn = getattr(user_module, 'input_fn', None)
            predict_fn = getattr(user_module, 'predict_fn', None)
            output_fn = getattr(user_module, 'output_fn', None)

            if transform_fn and (input_fn or predict_fn or output_fn):
                raise ValueError('Cannot use transform_fn implementation in conjunction with input_fn, predict_fn, '
                                 'and/or output_fn implementation')

            self._transform_fn = transform_fn or self._default_transform_fn
            self._input_fn = input_fn or self._default_inference_handler.default_input_fn
            self._predict_fn = predict_fn or self._default_inference_handler.default_predict_fn
            self._output_fn = output_fn or self._default_inference_handler.default_output_fn
        except Exception:
            self._model_fn = self._default_inference_handler.default_model_fn
            self._transform_fn = self._default_transform_fn
            self._input_fn = self._default_inference_handler.default_input_fn
            self._predict_fn = self._default_inference_handler.default_predict_fn
            self._output_fn = self._default_inference_handler.default_output_fn

    def _default_transform_fn(self, model, input_data, content_type, accept):
        """Make predictions against the model and return a serialized response.
        This serves as the default implementation of transform_fn, used when the user has not
        provided an implementation.
        Args:
            model (obj): model loaded by model_fn.
            input_data (obj): the request data.
            content_type (str): the request content type.
            accept (str): accept header expected by the client.
        Returns:
            obj:
                the serialized prediction result or a tuple of the form (response_data, content_type)
        """
        data = self._input_fn(input_data, content_type)
        prediction = self._predict_fn(data, model)
        result = self._output_fn(prediction, accept)
        return result
