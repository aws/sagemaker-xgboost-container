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

import os
import unittest
from pathlib import Path

from sagemaker_algorithm_toolkit.exceptions import UserError
from sagemaker_xgboost_container.data_utils import CSV, LIBSVM, PARQUET
from sagemaker_xgboost_container.distributed_gpu.dask_data_utils import _read_data


class TestDaskDataUtils(unittest.TestCase):
    NUM_ROWS_IN_EACH_FILE = 5
    NUM_COLS_IN_EACH_FILE = 6

    def setUp(self):
        current_path = Path(os.path.abspath(__file__))
        self.data_path_csv = os.path.join(
            str(current_path.parent.parent.parent), "resources", "data", "csv", "csv_files"
        )
        self.data_path_csv_multiple = os.path.join(
            str(current_path.parent.parent.parent), "resources", "data", "csv", "multiple_files"
        )
        self.data_path_parquet = os.path.join(
            str(current_path.parent.parent.parent), "resources", "data", "parquet", "multiple_files"
        )

    def test_read_data_csv(self):
        x, y = _read_data(self.data_path_csv, CSV)
        assert x.shape[0].compute() == self.NUM_ROWS_IN_EACH_FILE
        assert x.shape[1] == self.NUM_COLS_IN_EACH_FILE - 1
        assert y.shape[0].compute() == self.NUM_ROWS_IN_EACH_FILE

    def test_read_data_csv_malformed_path(self):
        x, y = _read_data(self.data_path_csv + "/", CSV)
        assert x.shape[0].compute() == self.NUM_ROWS_IN_EACH_FILE

    def test_read_data_csv_multiple_files(self):
        x, y = _read_data(self.data_path_csv_multiple, CSV)
        assert x.shape[0].compute() == self.NUM_ROWS_IN_EACH_FILE * 2

    def test_read_data_parquet(self):
        x, y = _read_data(self.data_path_parquet, PARQUET)
        assert x.shape[0].compute() == self.NUM_ROWS_IN_EACH_FILE * 2
        assert x.shape[1] == self.NUM_COLS_IN_EACH_FILE - 1
        assert y.shape[0].compute() == self.NUM_ROWS_IN_EACH_FILE * 2

    def test_read_data_unsupported_content(self):
        with self.assertRaises(UserError):
            _read_data(self.data_path_parquet, LIBSVM)
