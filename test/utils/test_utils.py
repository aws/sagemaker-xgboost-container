from __future__ import absolute_import

import test.utils.local_mode as localmode


def files_exist(opt_ml, files):
    for f in files:
        assert localmode.file_exists(opt_ml, f), 'file {} was not created'.format(f)


def predict_and_assert_response_length(data, content_type):
    predict_response = localmode.request(data, content_type=content_type)
    assert len(predict_response) == len(data)
