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

import unittest

from sagemaker_algorithm_toolkit import channel_validation as cv
from sagemaker_xgboost_container.algorithm_mode import channel_validation as acv

REQUIRED_CHANNEL = "required"
NOT_REQUIRED_CHANNEL = "not_required"


class TestChannelValidation(unittest.TestCase):
    def setUp(self):
        self.channels = acv.initialize()

    def test_default_content_type(self):
        test_user_channels = {"train": {cv.TRAINING_INPUT_MODE: "File", cv.S3_DIST_TYPE: "FullyReplicated"}}
        self.channels.validate(test_user_channels)
        self.assertEqual(test_user_channels["train"][cv.CONTENT_TYPE], "text/libsvm")
