from __future__ import absolute_import

import socket
from contextlib import closing

import test.utils.local_mode as localmode


def files_exist(opt_ml, files):
    for f in files:
        assert localmode.file_exists(opt_ml, f), 'file {} was not created'.format(f)


def predict_and_assert_response_length(data, content_type):
    predict_response = localmode.request(data, content_type=content_type)
    assert len(predict_response) == len(data)


# From https://stackoverflow.com/a/45690594
def find_two_open_ports():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s1:
        s1.bind(('', 0))
        s1.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s2:
            s2.bind(('', 0))
            s2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            return s1.getsockname()[1], s2.getsockname()[1]