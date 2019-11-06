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
import http.client


class BaseInferenceToolkitError(Exception):
    """Exception used to indicate a problem that occurred during inference.

    This is meant to be extended from so that customers may handle errors within inference servers.

    TODO: This should be moved to the sagemaker-inference-toolkit repo

    :param status_code: HTTP Error Status Code to send to client
    :param message: Response message to send to client
    :param phrase: Response body to send to client
    """
    def __init__(self, status_code, message, phrase):
        self.status_code = status_code
        self.message = message
        self.phrase = phrase


class NoContentInferenceError(BaseInferenceToolkitError):
    def __init__(self):
        super(NoContentInferenceError, self).__init__(http.client.NO_CONTENT, "", "")


class UnsupportedMediaTypeInferenceError(BaseInferenceToolkitError):
    def __init__(self, message):
        super(UnsupportedMediaTypeInferenceError, self).__init__(http.client.UNSUPPORTED_MEDIA_TYPE, message, message)


class ModelLoadInferenceError(BaseInferenceToolkitError):
    def __init__(self, message):
        formatted_message = "Unable to load model: {}".format(message)
        super(ModelLoadInferenceError, self).__init__(http.client.INTERNAL_SERVER_ERROR,
                                                      formatted_message,
                                                      formatted_message)


class BadRequestInferenceError(BaseInferenceToolkitError):
    def __init__(self, message):
        formatted_message = "Unable to evaluate payload provided: {}".format(message)
        super(BadRequestInferenceError, self).__init__(http.client.BAD_REQUEST,
                                                       formatted_message,
                                                       formatted_message)
