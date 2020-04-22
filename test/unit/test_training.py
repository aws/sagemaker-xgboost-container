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
from mock import MagicMock, patch
from sagemaker_xgboost_container import training


def mock_training_env(current_host='algo-1', module_dir='s3://my/script', module_name='svm', **kwargs):
    return MagicMock(current_host=current_host, module_dir=module_dir, module_name=module_name, **kwargs)


class TestTraining(unittest.TestCase):
    """Note: The 'train' method has been mocked since this test only checks the training resource setup"""

    @patch('sagemaker_training.entry_point.run')
    def test_script_mode(self, mock_run_module):
        env = mock_training_env()
        env.user_entry_point = "dummy_entry_point"
        training.train(env)

        mock_run_module.assert_called_with(uri='s3://my/script',
                                           user_entry_point='svm',
                                           args=env.to_cmd_args(),
                                           env_vars=env.to_env_vars(),
                                           capture_error=False)

    @patch('sagemaker_xgboost_container.training.run_algorithm_mode')
    def test_algorithm_mode(self, mock_algorithm_mode_train):
        env = mock_training_env(module_dir="")
        env.user_entry_point = None
        training.train(env)

        mock_algorithm_mode_train.assert_called_with()
