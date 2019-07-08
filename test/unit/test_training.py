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
import os

from sagemaker_xgboost_container import training
from sagemaker_xgboost_container import distributed


test_master = 'test_master'
test_slave = 'test_slave'
test_hosts = [test_master, test_slave]


def mock_training_env(current_host='algo-1', module_dir='s3://my/script', module_name='svm', **kwargs):
    return MagicMock(current_host=current_host, module_dir=module_dir, module_name=module_name, **kwargs)


def get_rabit_message(host_name, has_data_boolean=True):
    return {'host': host_name, 'has_data': has_data_boolean}


class MockRabit(distributed.Rabit):
    """Dummy Rabit class to use in order to avoid mocking the socket connections
    """
    def __init__(self, mock_rabit_helper=None):
        if mock_rabit_helper:
            self.rabit_helper = mock_rabit_helper
        else:
            self.rabit_helper = MagicMock()

    def __enter__(self):
        return self.rabit_helper

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass


class TestMapReader(unittest.TestCase):
    """Note: The 'train' method has been mocked since this test only checks the training resource setup"""

    @patch('sagemaker_containers.beta.framework.modules.run_module')
    def test_script_mode(self, mock_run_module):
        env = mock_training_env()
        env.user_entry_point = "dummy_entry_point"
        training.train(env)

        mock_run_module.assert_called_with('s3://my/script', env.to_cmd_args(), env.to_env_vars(), 'svm')

    @patch('sagemaker_xgboost_container.algorithm_mode.default_entry_point.algorithm_mode_train')
    def test_algorithm_mode(self, mock_algorithm_mode_train):
        env = mock_training_env(module_dir="")
        env.user_entry_point = None
        training.train(env)

        mock_algorithm_mode_train.assert_called_with(True)

    @patch('os.listdir')
    def test_has_train_data(self, mock_os_listdir):
        mock_os_listdir.return_value = ['dummy_file_1', 'dummy_file_2']
        self.assertTrue(training._has_train_data('dummy_path'))

    def _test_setup_rabit_and_train(self,
                                    current_host,
                                    current_host_train_files,
                                    other_host_messages):
        with patch('os.listdir') as mock_os_listdir, \
                patch.dict('os.environ', {'SM_CHANNEL_TRAIN': 'train_path'}), \
                patch('sagemaker_xgboost_container.distributed.Rabit') as mock_rabit, \
                patch('sagemaker_xgboost_container.training.train') as mock_train:

            # Create list of synchronized 'has_data' calls
            mock_os_listdir.return_value = current_host_train_files
            current_host_message = get_rabit_message(current_host, training._has_train_data())
            all_host_messages = other_host_messages + [current_host_message]

            # Set up mocks for both Rabit initializations
            mock_first_rabit_helper = MagicMock()
            mock_first_rabit_helper.current_host = current_host
            mock_first_rabit_helper.synchronize.return_value = all_host_messages

            mock_second_rabit_helper = MagicMock()
            mock_second_rabit_helper.is_master = current_host == test_master
            mock_rabit.side_effect = [MockRabit(mock_first_rabit_helper), MockRabit(mock_second_rabit_helper)]

            env = mock_training_env()
            env.user_entry_point = None

            if not current_host_message['has_data']:
                # If current host has no data, initialize Rabit once then SystemExit
                with self.assertRaises(SystemExit):
                    training.setup_rabit_and_train(env, test_hosts, current_host)

                    mock_first_rabit_helper.synchronize.assert_called_once_with(current_host_message)
                    self.assertFalse(mock_train.called)
                    self.assertIsNone(os.environ.get('SM_IS_RABIT_MASTER', None))

            if current_host_message['has_data']:
                # If current host has data
                training.setup_rabit_and_train(env, test_hosts, current_host)
                mock_first_rabit_helper.synchronize.assert_called_once_with(current_host_message)
                mock_train.assert_called_once()

                if len(all_host_messages) > 1:
                    self.assertEqual(os.environ['SM_IS_RABIT_MASTER'], str(current_host == test_master))
                else:
                    self.assertIsNone(os.environ.get('SM_IS_RABIT_MASTER', None))

    def test_setup_rabit_and_train_two_hosts_with_data_as_master(self):
        self._test_setup_rabit_and_train(current_host=test_master,
                                         current_host_train_files=['dummy_file_1', 'dummy_file_2'],
                                         other_host_messages=[get_rabit_message(test_slave)])

    def test_setup_rabit_and_train_two_hosts_with_data_as_slave(self):
        self._test_setup_rabit_and_train(current_host=test_slave,
                                         current_host_train_files=['dummy_file_1', 'dummy_file_2'],
                                         other_host_messages=[get_rabit_message(test_slave)])

    def test_setup_rabit_and_train_current_host_with_data_only(self):
        self._test_setup_rabit_and_train(current_host=test_master,
                                         current_host_train_files=['dummy_file_1', 'dummy_file_2'],
                                         other_host_messages=[])

    def test_setup_rabit_and_train_current_host_no_data(self):
        self._test_setup_rabit_and_train(current_host=test_slave,
                                         current_host_train_files=[],
                                         other_host_messages=[get_rabit_message(test_master)])
