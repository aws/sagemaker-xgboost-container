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
from sagemaker_algorithm_toolkit import exceptions as exc


class Channel(object):
    """Represents a single SageMaker training job channel."""
    FILE_MODE = "File"
    PIPE_MODE = "Pipe"
    AUGMENTED_MODE = "Augmented"

    SHARDED = "Sharded"
    REPLICATED = "FullyReplicated"

    def __init__(self, name, required):
        self.name = name
        self.required = required
        self.supported = set()

    def format(self):
        """Formats channel for SageMaker's CreateAlgorithm API."""
        supported_content_types = list(set(c[0] for c in self.supported))
        supported_input_modes = list(set(c[1] for c in self.supported))
        return {"Name": self.name,
                "Description": self.name,
                "IsRequired": self.required,
                "SupportedContentTypes": supported_content_types,
                "SupportedInputModes": supported_input_modes,
                }

    def add(self, content_type, supported_input_mode, supported_s3_data_distribution_type):
        """Adds relevant configuration as a supported configuration for the channel."""
        self.supported.add((content_type, supported_input_mode, supported_s3_data_distribution_type))

    def validate(self, value):
        """Validates the provided configuration against the channel's supported configuration."""
        if (value["ContentType"], value["TrainingInputMode"], value["S3DistributionType"]) not in self.supported:
            raise exc.UserError("Channel configuration for '{}' channel is not supported: {}".format(self.name, value))


class Channels(object):
    """Represents a collection of Channels for a SageMaker training job."""

    def __init__(self, *channels):
        self.channels = channels

    def format(self):
        """Formats channels for SageMaker's CreateAlgorithm API."""
        return [channel.format() for channel in self.channels]

    def validate(self, user_channels):
        """Validates the provided user-specified channels at runtime against the channels' supported configuration."""
        for channel in self.channels:
            if channel.name not in user_channels:
                if channel.required:
                    raise exc.UserError("Missing required channel: {}".format(channel.name))

        name_to_channel = {channel.name: channel for channel in self.channels}
        validated_channels = {}
        for channel, value in user_channels.items():
            try:
                channel_obj = name_to_channel[channel]
            except KeyError:
                raise exc.UserError("Extraneous channel found: {}".format(channel_obj))
            channel_obj.validate(value)
            validated_channels[channel] = value

        return validated_channels
