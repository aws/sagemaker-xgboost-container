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
