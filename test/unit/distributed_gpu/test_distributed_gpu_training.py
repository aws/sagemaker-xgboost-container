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
from sagemaker_xgboost_container.distributed_gpu.distributed_gpu_training import (
    INPUT_FORMAT_ERROR_MSG,
    NON_GPU_ERROR_MSG,
    NOT_REPLICATED_ERROR_MSG,
    PIPE_MODE_ERROR_MSG,
    validate_gpu_train_configuration,
)


class TestDistributedGPUTraining(unittest.TestCase):
    def setUp(self):
        self.train_channel_replicated = {"train": {cv.S3_DIST_TYPE: "FullyReplicated"}}
        self.train_channel_not_replicated = {"train": {cv.S3_DIST_TYPE: "ShardedByS3Key"}}
        self.multi_channel_not_replicated = {
            "train": {cv.S3_DIST_TYPE: "FullyReplicated"},
            "valid": {cv.S3_DIST_TYPE: "ShardedByS3Key"},
        }

    def test_conditions_fail_channel_not_replicated_multi_host(self):
        exc_list = validate_gpu_train_configuration("gpu_hist", 2, 1, "File", "csv", self.multi_channel_not_replicated)
        assert NOT_REPLICATED_ERROR_MSG in exc_list

    def test_conditions_pass_channel_replicated_multi_host(self):
        exc_list = validate_gpu_train_configuration("gpu_hist", 2, 1, "File", "csv", self.train_channel_replicated)
        assert not exc_list

    def test_conditions_pass_channel_not_replicated_singlehost(self):
        exc_list = validate_gpu_train_configuration("gpu_hist", 1, 1, "File", "csv", self.train_channel_not_replicated)
        assert not exc_list

    def test_conditions_fail_not_gpu_instance(self):
        exc_list = validate_gpu_train_configuration("gpu_hist", 1, 0, "File", "csv", self.train_channel_replicated)
        assert NON_GPU_ERROR_MSG in exc_list

    def test_conditions_fail_non_gpu_tree_method(self):
        exc_list = validate_gpu_train_configuration("approx", 1, 1, "File", "csv", self.train_channel_replicated)
        assert NON_GPU_ERROR_MSG in exc_list

    def test_conditions_fail_pipe_mode(self):
        exc_list = validate_gpu_train_configuration("gpu_hist", 1, 1, "Pipe", "csv", self.train_channel_replicated)
        assert PIPE_MODE_ERROR_MSG in exc_list

    def test_conditions_fail_unsupported_format(self):
        exc_list = validate_gpu_train_configuration("gpu_hist", 1, 1, "File", "libsvm", self.train_channel_replicated)
        assert INPUT_FORMAT_ERROR_MSG in exc_list

    def test_conditions_fail_multiple_checks(self):
        exc_list = validate_gpu_train_configuration("approx", 1, 1, "Pipe", "libsvm", self.train_channel_replicated)
        assert len(exc_list) == 3
