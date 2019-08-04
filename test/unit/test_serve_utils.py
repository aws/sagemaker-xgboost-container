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
from mock import Mock
import unittest

from sagemaker_algorithm_toolkit import exceptions
from sagemaker_xgboost_container.algorithm_mode import serve_utils


class TestServeUtils(unittest.TestCase):
    def test_get_content_type_libsvm(self):
        request = Mock()
        request.content_type = "text/libsvm"
        content_type = serve_utils.get_content_type(request)
        self.assertEqual(content_type, "text/libsvm")

    def test_get_content_type_empty(self):
        request = Mock()
        request.content_type = None
        content_type = serve_utils.get_content_type(request)
        self.assertEqual(content_type, "text/csv")

    def test_get_content_type_invalid(self):
        request = Mock()
        request.content_type = "invalid/type"
        self.assertRaises(exceptions.UserError, serve_utils.get_content_type, request)

    def test_get_content_type_serial_pipeline(self):
        request = Mock()
        request.content_type = "text/csv;charset=utf-8"
        content_type = serve_utils.get_content_type(request)
        self.assertEqual(content_type, "text/csv")
