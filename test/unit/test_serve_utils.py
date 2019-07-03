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
