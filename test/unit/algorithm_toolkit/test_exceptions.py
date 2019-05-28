import unittest

from sagemaker_algorithm_toolkit import exceptions as exc


class TestExceptions(unittest.TestCase):
    def test_BaseToolkitError(self):
        e = exc.BaseToolkitError()
        self.assertEqual(e.message, "unknown error occurred")

    def test_BaseToolkitError_ValueError(self):
        e = exc.BaseToolkitError(caused_by=ValueError("abc"))
        self.assertEqual(e.message, "abc (caused by ValueError)")

    def test_UserError(self):
        e = exc.UserError("Test 123")
        self.assertEqual(e.message, "Test 123")

    def test_UserError_ValueError(self):
        e = exc.UserError("Test 123", caused_by=ValueError("abc"))
        self.assertEqual(e.message, "Test 123 (caused by ValueError)")

    def test_AlgorithmError(self):
        e = exc.AlgorithmError("Test 123")
        self.assertEqual(e.message, "Test 123")

    def test_AlgorithmError_ValueError(self):
        e = exc.AlgorithmError("Test 123", caused_by=ValueError("abc"))
        self.assertEqual(e.message, "Test 123 (caused by ValueError)")

    def test_PlatformError(self):
        e = exc.PlatformError("Test 123")
        self.assertEqual(e.message, "Test 123")

    def test_PlatformError_ValueError(self):
        e = exc.PlatformError("Test 123", caused_by=ValueError("abc"))
        self.assertEqual(e.message, "Test 123 (caused by ValueError)")
