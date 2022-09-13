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

import sys
import time
from multiprocessing import Process, Queue
from test.utils.test_utils import find_two_open_ports

import pytest

from sagemaker_xgboost_container import distributed


def synchronize_fn(host_count, port, master, idx, q):
    hosts = ["127.0.0.1"] + ["localhost" for _ in range(host_count - 1)]
    current_host = "127.0.0.1" if master else "localhost"
    with distributed.Rabit(hosts, current_host=current_host, port=port, master_host="127.0.0.1") as dr:
        results = dr.synchronize({"idx": idx})
    q.put(results)
    sys.exit(0)


def rabit_run_fn(
    host_count, is_run, first_port, second_port, master, idx, q, max_connect_attempts=None, connect_retry_timeout=3
):
    hosts = ["127.0.0.1"] + ["localhost" for _ in range(host_count - 1)]
    current_host = "127.0.0.1" if master else "localhost"
    args_dict = dict(obj=idx)

    distributed.rabit_run(
        q.put,
        args_dict,
        is_run,
        hosts,
        current_host,
        first_port,
        second_port,
        max_connect_attempts=max_connect_attempts,
        connect_retry_timeout=connect_retry_timeout,
        update_rabit_args=False,
    )

    sys.exit(0)


def rabit_run_delay_master(host_count, is_run, first_port, second_port, master, idx, q, max_connect_attempts):
    if master:
        time.sleep(10)

    rabit_run_fn(host_count, is_run, first_port, second_port, master, idx, q, max_connect_attempts=max_connect_attempts)


def rabit_run_fail(test_fn, host_count, is_run, first_port, second_port, master, idx, q, max_connect_attempts=None):
    try:
        test_fn(host_count, is_run, first_port, second_port, master, idx, q, max_connect_attempts=max_connect_attempts)

        raise Exception("This rabit run should fail!")
    except Exception as e:
        q.put("{} {}".format(idx, str(e)))


def test_integration_rabit_synchronize():
    q = Queue()

    port, _ = find_two_open_ports()

    host_count = 5
    host_list = range(host_count)
    expected_results = [{"idx": idx} for idx in host_list]

    for idx in host_list:
        p = Process(target=synchronize_fn, args=(host_count, port, idx == 0, idx, q))
        p.start()

    num_responses = 0
    while num_responses < host_count:
        host_aggregated_result = q.get(timeout=10)
        for host_individual_result in host_aggregated_result:
            assert host_individual_result in expected_results
        num_responses += 1


def test_rabit_run_all_hosts_run():
    q = Queue()

    first_port, second_port = find_two_open_ports()

    host_count = 5
    host_list = range(host_count)
    expected_results = [idx for idx in host_list]

    for idx in host_list:
        p = Process(target=rabit_run_fn, args=(host_count, True, first_port, second_port, idx == 0, idx, q))
        p.start()

    num_responses = 0
    while num_responses < host_count:
        response = q.get(timeout=15)
        expected_results.remove(response)
        num_responses += 1

    assert len(expected_results) == 0


def test_rabit_run_exclude_one_host():
    q = Queue()

    first_port, second_port = find_two_open_ports()

    idx_to_exclude = 3

    host_count = 5
    host_list = range(host_count)
    expected_results = [idx for idx in host_list if idx != idx_to_exclude]

    for idx in host_list:
        p = Process(
            target=rabit_run_fn, args=(host_count, idx != idx_to_exclude, first_port, second_port, idx == 0, idx, q)
        )
        p.start()

    num_responses = 0
    while num_responses < host_count - 1:
        response = q.get(timeout=15)
        expected_results.remove(response)
        num_responses += 1

    assert len(expected_results) == 0


def test_rabit_delay_master():
    q = Queue()

    first_port, second_port = find_two_open_ports()

    host_count = 5
    host_list = range(host_count)
    expected_results = [idx for idx in host_list]

    for idx in host_list:
        p = Process(
            target=rabit_run_delay_master, args=(host_count, True, first_port, second_port, idx == 0, idx, q, None)
        )
        p.start()

    num_responses = 0
    while num_responses < host_count:
        response = q.get(timeout=20)
        expected_results.remove(response)
        num_responses += 1

    assert len(expected_results) == 0


@pytest.mark.parametrize("bad_max_retry_attempts", [0, -1])
def test_rabit_run_fail_bad_max_retry_attempts(bad_max_retry_attempts):
    q = Queue()

    first_port, second_port = find_two_open_ports()

    host_count = 5
    host_list = range(host_count)

    for idx in host_list:
        p = Process(
            target=rabit_run_fail,
            args=(rabit_run_fn, host_count, True, first_port, second_port, idx == 0, idx, q, bad_max_retry_attempts),
        )
        p.start()

    num_responses = 0
    while num_responses < host_count:
        host_result = q.get(timeout=10)
        assert "max_connect_attempts must be None or an integer greater than 0." in host_result
        num_responses += 1
