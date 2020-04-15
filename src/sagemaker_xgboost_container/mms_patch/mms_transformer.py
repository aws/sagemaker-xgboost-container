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

import http
import logging

from sagemaker_inference import content_types, utils
from sagemaker_inference.errors import BaseInferenceToolkitError
from sagemaker_inference.transformer import Transformer


class XGBMMSTransformer(Transformer):

    def transform(self, data, context):
        """Take a request with input data, deserialize it, make a prediction, and return a
        serialized response.

        NOTE: This is almost a copy of the original Transformer method, except it does not decode the utf-8 data.
        This is done for backwards compatibility.

        See line removed here:
        https://github.com/aws/sagemaker-inference-toolkit/blob/master/src/sagemaker_inference/transformer.py#L123

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
                return self.handle_error(context, http.HTTPStatus.BAD_REQUEST, e.message)
