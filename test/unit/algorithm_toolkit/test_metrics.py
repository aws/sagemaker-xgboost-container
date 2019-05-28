from sagemaker_algorithm_toolkit import metrics as m

import mock
import unittest


class TestMetrics(unittest.TestCase):
    def test_simple(self):
        metrics = m.Metrics(m.Metric(name="test mean squared error",
                                     format_string="test:mse {:.3f}",
                                     direction=m.Metric.MINIMIZE,
                                     regex="test:mse ([0-9\\.]+)"))
        with mock.patch("sagemaker_algorithm_toolkit.metrics.logging") as mock_logging:
            metrics["test mean squared error"].log(5.123)
            mock_logging.info.assert_called_once_with("test:mse 5.123")
