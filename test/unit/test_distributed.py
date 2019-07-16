from __future__ import absolute_import

import sys

from test.utils.test_utils import find_two_open_ports
from multiprocessing import Process, Queue

from sagemaker_xgboost_container import distributed


def synchronize_fn(host_count, port, master, idx, q):
    hosts = ['127.0.0.1'] + ['localhost' for _ in range(host_count - 1)]
    current_host = '127.0.0.1' if master else 'localhost'
    with distributed.Rabit(hosts, current_host=current_host, port=port, master_host='127.0.0.1') as dr:
        results = dr.synchronize({
            'idx': idx
        })
    q.put(results)
    sys.exit(0)


def rabit_run_fn(host_count, is_run, first_port, second_port, master, idx, q):
    hosts = ['127.0.0.1'] + ['localhost' for _ in range(host_count - 1)]
    current_host = '127.0.0.1' if master else 'localhost'
    args_dict = dict(obj=idx)

    distributed.rabit_run(
        q.put, args_dict, is_run, hosts, current_host, first_port, second_port, update_rabit_args=False)
    sys.exit(0)


def test_integration_rabit_synchronize():
    q = Queue()

    port, _ = find_two_open_ports()

    host_count = 5
    host_list = range(host_count)
    expected_results = [{'idx': idx} for idx in host_list]

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
        p = Process(target=rabit_run_fn, args=(
            host_count, idx != idx_to_exclude, first_port, second_port, idx == 0, idx, q))
        p.start()

    num_responses = 0
    while num_responses < host_count - 1:
        response = q.get(timeout=15)
        expected_results.remove(response)
        num_responses += 1

    assert len(expected_results) == 0
