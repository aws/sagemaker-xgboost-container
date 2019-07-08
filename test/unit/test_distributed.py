# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import logging
import mock
import os
import pytest
import tempfile

from xgboost import rabit

from sagemaker_xgboost_container import distributed
from sagemaker_xgboost_container.dmlc_patch import tracker

test_master = 'test_host_1'
test_slave = 'test_host_2'
test_hosts = [test_master, test_slave, 'test_host_3']
test_rank = 1


class MockSocket():
    def __init__(self, raise_exception=False):
        """Helper class to mock socket connections.

        :param raise_exception: Simulate OSError when server is not accepting connections
        """
        self.raise_exception = raise_exception

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def connect(self, address):
        if self.raise_exception:
            raise OSError()
        else:
            pass


def test_rabit_tracker_initialize():
    test_cluster = distributed.Rabit(hosts=test_hosts)
    assert test_cluster.master_host == test_master
    assert not test_cluster.is_master_host
    assert test_cluster.current_host == distributed.LOCAL_HOSTNAME


def _get_rabit_init_message(n_workers, master_host, port):
    return ['DMLC_NUM_WORKER={}'.format(n_workers).encode(),
            'DMLC_TRACKER_URI={}'.format(master_host).encode(),
            'DMLC_TRACKER_PORT={}'.format(port).encode()]


@mock.patch('xgboost.rabit.init')
@mock.patch('xgboost.rabit.get_rank')
@mock.patch('socket.socket')
def test_slave_start(mock_socket, mock_xgb_rabit_get_rank, mock_xgb_rabit_init):
    test_cluster = distributed.Rabit(hosts=test_hosts, current_host=test_slave)
    assert not test_cluster.is_master_host

    mock_socket.return_value = MockSocket()
    mock_xgb_rabit_get_rank.return_value = test_rank

    test_rabit_helper = test_cluster.start()
    assert not test_rabit_helper.is_master
    assert test_rabit_helper.rank == test_rank
    assert test_rabit_helper.current_host == test_slave
    assert test_rabit_helper.master_port == 9099

    mock_xgb_rabit_init.assert_called_with(_get_rabit_init_message(len(test_hosts),
                                                                   test_master,
                                                                   9099))


@mock.patch('sagemaker_xgboost_container.dmlc_patch.tracker.RabitTracker')
@mock.patch('xgboost.rabit.init')
@mock.patch('xgboost.rabit.get_rank')
@mock.patch('socket.socket')
def test_master_start(mock_socket, mock_xgb_rabit_get_rank, mock_xgb_rabit_init, mock_rabit_tracker):
    test_cluster = distributed.Rabit(hosts=test_hosts, current_host=test_master)
    assert test_cluster.is_master_host

    mock_rabit_context = mock.Mock()

    mock_socket.return_value = MockSocket()
    mock_xgb_rabit_get_rank.return_value = test_rank
    mock_rabit_tracker.return_value = mock_rabit_context

    test_rabit_helper = test_cluster.start()
    assert test_rabit_helper.is_master
    assert test_rabit_helper.rank == test_rank
    assert test_rabit_helper.current_host == test_master
    assert test_rabit_helper.master_port == 9099

    mock_xgb_rabit_init.assert_called_with(_get_rabit_init_message(len(test_hosts),
                                                                   test_master,
                                                                   9099))
    mock_rabit_context.start.assert_called_with(len(test_hosts))


@mock.patch('xgboost.rabit.finalize')
@mock.patch('socket.socket')
def test_slave_stop(mock_socket, mock_xgb_rabit_finalize):
    test_cluster = distributed.Rabit(hosts=test_hosts, current_host=test_slave)
    assert not test_cluster.is_master_host

    mock_socket.return_value = MockSocket(raise_exception=True)
    test_cluster.stop()


@mock.patch('xgboost.rabit.finalize')
@mock.patch('socket.socket')
def test_master_start(mock_socket, mock_xgb_rabit_finalize):
    test_cluster = distributed.Rabit(hosts=test_hosts, current_host=test_master)
    assert test_cluster.is_master_host

    mock_rabit_context = mock.Mock()
    test_cluster.rabit_context = mock_rabit_context

    mock_socket.return_value = MockSocket(raise_exception=True)
    test_cluster.stop()

    mock_rabit_context.join.assert_called_once_with()
