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

from mock import patch
from mock import MagicMock
# import numpy as np
# import pytest
# import xgboost as xgb
#
# from sagemaker_containers.beta.framework import (content_types, encoders, errors)
from sagemaker_xgboost_container import serving


@patch('sagemaker_inference.model_server.start_model_server')
def test_hosting_start(start_model_server):
    environ = MagicMock()
    start_response = MagicMock()
    serving.main() # environ, start_response)
    start_model_server.assert_called_with(
        handler_service='sagemaker_xgboost_container.algorithm_mode.handler_service')
