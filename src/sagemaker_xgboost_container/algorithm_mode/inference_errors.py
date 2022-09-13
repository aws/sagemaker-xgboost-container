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

from sagemaker_inference import errors


class NoContentInferenceError(errors.BaseInferenceToolkitError):
    def __init__(self):
        super(NoContentInferenceError, self).__init__(http.client.NO_CONTENT, "", "")


class UnsupportedMediaTypeInferenceError(errors.BaseInferenceToolkitError):
    def __init__(self, message):
        super(UnsupportedMediaTypeInferenceError, self).__init__(http.client.UNSUPPORTED_MEDIA_TYPE, message, message)


class ModelLoadInferenceError(errors.BaseInferenceToolkitError):
    def __init__(self, message):
        formatted_message = "Unable to load model: {}".format(message)
        super(ModelLoadInferenceError, self).__init__(
            http.client.INTERNAL_SERVER_ERROR, formatted_message, formatted_message
        )


class BadRequestInferenceError(errors.BaseInferenceToolkitError):
    def __init__(self, message):
        formatted_message = "Unable to evaluate payload provided: {}".format(message)
        super(BadRequestInferenceError, self).__init__(http.client.BAD_REQUEST, formatted_message, formatted_message)
