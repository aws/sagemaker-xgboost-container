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
import unittest

from sagemaker_algorithm_toolkit import channel_validation as cv
from sagemaker_algorithm_toolkit import exceptions as exc


class TestChannelValidation(unittest.TestCase):
    def test_simple_supported(self):
        channel = cv.Channel(name="train", required=True)
        channel.add("text/csv", cv.Channel.FILE_MODE, cv.Channel.REPLICATED)
        channels = cv.Channels(channel)
        channels.validate(
            {
                "train": {
                    cv.CONTENT_TYPE: "text/csv",
                    cv.TRAINING_INPUT_MODE: "File",
                    cv.S3_DIST_TYPE: "FullyReplicated",
                    "RecordWrapperType": "None",
                }
            }
        )

    def test_default_content_type(self):
        channel = cv.Channel(name="train", required=True)
        channel.add("text/csv", cv.Channel.FILE_MODE, cv.Channel.REPLICATED)
        channels = cv.Channels(channel)
        channels.set_default_content_type("text/csv")
        test_user_channels = {
            "train": {cv.TRAINING_INPUT_MODE: "File", cv.S3_DIST_TYPE: "FullyReplicated", "RecordWrapperType": "None"}
        }
        channels.validate(test_user_channels)
        self.assertEqual(test_user_channels["train"][cv.CONTENT_TYPE], "text/csv")

    def test_simple_not_supported(self):
        channel = cv.Channel(name="train", required=True)
        channel.add("text/csv", cv.Channel.FILE_MODE, cv.Channel.REPLICATED)
        channels = cv.Channels(channel)
        with self.assertRaises(exc.UserError):
            channels.validate(
                {
                    "train": {
                        cv.CONTENT_TYPE: "text/csv",
                        cv.TRAINING_INPUT_MODE: "Pipe",
                        cv.S3_DIST_TYPE: "FullyReplicated",
                        "RecordWrapperType": "None",
                    }
                }
            )

    def test_simple_extra(self):
        channel = cv.Channel(name="train", required=True)
        channel.add("text/csv", cv.Channel.FILE_MODE, cv.Channel.REPLICATED)
        channels = cv.Channels(channel)
        with self.assertRaises(exc.UserError):
            channels.validate(
                {
                    "train": {
                        cv.CONTENT_TYPE: "text/csv",
                        cv.TRAINING_INPUT_MODE: "File",
                        cv.S3_DIST_TYPE: "FullyReplicated",
                        "RecordWrapperType": "None",
                    },
                    "extra": {},
                }
            )

    def test_simple_required(self):
        channel = cv.Channel(name="train", required=True)
        channel.add("text/csv", cv.Channel.FILE_MODE, cv.Channel.REPLICATED)
        channels = cv.Channels(channel)
        with self.assertRaises(exc.UserError):
            channels.validate({"sorry": {}})

    def test_simple_format(self):
        channel = cv.Channel(name="train", required=True)
        channel.add("text/csv", cv.Channel.FILE_MODE, cv.Channel.REPLICATED)
        channels = cv.Channels(channel)

        result = {
            "Name": "train",
            "Description": "train",
            "IsRequired": True,
            "SupportedContentTypes": ["text/csv"],
            "SupportedInputModes": ["File"],
        }
        self.assertEqual(channel.format(), result)
        self.assertEqual(channels.format(), [result])
