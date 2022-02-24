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
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from sagemaker_algorithm_toolkit import exceptions as exc
from sagemaker_xgboost_container import data_utils


class TestTrainUtils(unittest.TestCase):

    def setUp(self):
        current_path = Path(os.path.abspath(__file__))
        self.data_path = os.path.join(str(current_path.parent.parent), 'resources', 'data')
        self.utils_path = os.path.join(str(current_path.parent.parent), 'utils')

    def test_get_content_type(self):
        self.assertEqual('libsvm', data_utils.get_content_type('libsvm'))
        self.assertEqual('libsvm', data_utils.get_content_type('text/libsvm'))
        self.assertEqual('libsvm', data_utils.get_content_type('text/x-libsvm'))

        self.assertEqual('csv', data_utils.get_content_type('csv'))
        self.assertEqual('csv', data_utils.get_content_type('text/csv'))
        self.assertEqual('csv', data_utils.get_content_type('text/csv; label_size=1'))
        self.assertEqual('csv', data_utils.get_content_type('text/csv;label_size = 1'))
        self.assertEqual('csv', data_utils.get_content_type('text/csv; charset=utf-8'))
        self.assertEqual('csv', data_utils.get_content_type('text/csv; label_size=1; charset=utf-8'))

        self.assertEqual('parquet', data_utils.get_content_type('parquet'))
        self.assertEqual('parquet', data_utils.get_content_type('application/x-parquet'))

        self.assertEqual('recordio-protobuf', data_utils.get_content_type('recordio-protobuf'))
        self.assertEqual('recordio-protobuf', data_utils.get_content_type('application/x-recordio-protobuf'))

        with self.assertRaises(exc.UserError):
            data_utils.get_content_type('incorrect_format')
        with self.assertRaises(exc.UserError):
            data_utils.get_content_type('text/csv; label_size=5')
        with self.assertRaises(exc.UserError):
            data_utils.get_content_type('text/csv; label_size=1=1')
        with self.assertRaises(exc.UserError):
            data_utils.get_content_type('text/csv; label_size=1; label_size=2')
        with self.assertRaises(exc.UserError):
            data_utils.get_content_type('label_size=1; text/csv')

    def test_validate_csv_files(self):
        csv_file_paths = ['train.csv', 'train.csv.weights', 'csv_files']

        for file_path in csv_file_paths:
            with self.subTest(file_path=file_path):
                csv_path = os.path.join(self.data_path, 'csv', file_path)
                data_utils.validate_data_file_path(csv_path, 'csv')

    def test_validate_libsvm_files(self):
        libsvm_file_paths = ['train.libsvm', 'train.libsvm.weights', 'libsvm_files']

        for file_path in libsvm_file_paths:
            with self.subTest(file_path=file_path):
                csv_path = os.path.join(self.data_path, 'libsvm', file_path)
                data_utils.validate_data_file_path(csv_path, 'libsvm')

    def _check_dmatrix(self, reader, path, num_col, num_row, *args):
        single_node_dmatrix = reader(path, *args)

        self.assertEqual(num_col, single_node_dmatrix.num_col())
        self.assertEqual(num_row, single_node_dmatrix.num_row())

        # no_weight_test_features = ["f{}".format(idx) for idx in range(single_node_dmatrix.num_col())]

        # self.assertEqual(no_weight_test_features, single_node_dmatrix.feature_names)

    def _check_piped_dmatrix(self, file_path, pipe_dir, reader, num_col, num_row, *args):
        python_exe = sys.executable
        pipe_cmd = '{}/sagemaker_pipe.py train {} {}'.format(self.utils_path, file_path, pipe_dir)

        proc = subprocess.Popen([python_exe] + pipe_cmd.split(" "))

        try:
            time.sleep(1)
            pipe_path = os.path.join(pipe_dir, 'train')
            self._check_dmatrix(reader, pipe_path, num_col, num_row, *args)
        finally:
            os.kill(proc.pid, signal.SIGTERM)
            shutil.rmtree(pipe_dir)

    def _check_piped_dmatrix2(self, file_path, pipe_dir, reader, num_col, num_row, *args):
        python_exe = sys.executable
        pipe_cmd = '{}/sagemaker_pipe.py train {} {}'.format(self.utils_path, file_path, pipe_dir)
        pipe_cmd2 = '{}/sagemaker_pipe.py validation {} {}'.format(self.utils_path, file_path, pipe_dir)

        proc = subprocess.Popen([python_exe] + pipe_cmd.split(" "))
        proc2 = subprocess.Popen([python_exe] + pipe_cmd2.split(" "))

        try:
            time.sleep(1)
            pipes_path = [os.path.join(pipe_dir, 'train'), os.path.join(pipe_dir, 'validation')]
            self._check_dmatrix(reader, pipes_path, num_col, 2*num_row, *args)
        finally:
            os.kill(proc.pid, signal.SIGTERM)
            os.kill(proc2.pid, signal.SIGTERM)
            shutil.rmtree(pipe_dir)

    def test_get_dmatrix(self):
        current_path = Path(os.path.abspath(__file__))
        data_path = os.path.join(str(current_path.parent.parent), 'resources', 'abalone', 'data')
        file_path = [os.path.join(data_path, path) for path in ['train', 'validation']]

        dmatrix = data_utils.get_dmatrix(file_path, 'libsvm', 0, False)

        self.assertEqual(9, dmatrix.num_col())
        self.assertEqual(3548, dmatrix.num_row())

    def test_parse_csv_dmatrix(self):
        csv_file_paths_and_weight = [('train.csv', 0), ('train.csv.weights', 1), ('csv_files', 0)]

        for file_path, csv_weight in csv_file_paths_and_weight:
            with self.subTest(file_path=file_path, csv_weight=csv_weight):
                csv_path = os.path.join(self.data_path, 'csv', file_path)
                reader = data_utils.get_csv_dmatrix
                self._check_dmatrix(reader, csv_path, 5, 5, csv_weight)

    def test_parse_csv_dmatrix_pipe(self):
        csv_file_paths_and_weight = [('csv_files', 0), ('weighted_csv_files', 1)]

        for file_path, csv_weight in csv_file_paths_and_weight:
            with self.subTest(file_path=file_path, csv_weight=csv_weight):
                csv_path = os.path.join(self.data_path, 'csv', file_path)
                pipe_dir = os.path.join(self.data_path, 'csv', 'pipe_path', file_path)
                reader = data_utils.get_csv_dmatrix
                is_pipe = True
                self._check_piped_dmatrix(csv_path, pipe_dir, reader, 5, 5, csv_weight, is_pipe)

    def test_parse_csv_dmatrix_pipe2(self):
        csv_file_paths_and_weight = [('csv_files', 0), ('weighted_csv_files', 1)]

        for file_path, csv_weight in csv_file_paths_and_weight:
            with self.subTest(file_path=file_path, csv_weight=csv_weight):
                csv_path = os.path.join(self.data_path, 'csv', file_path)
                pipe_dir = os.path.join(self.data_path, 'csv', 'pipe_path2', file_path)
                reader = data_utils.get_csv_dmatrix
                is_pipe = True
                self._check_piped_dmatrix2(csv_path, pipe_dir, reader, 5, 5, csv_weight, is_pipe)

    def test_parse_libsvm_dmatrix(self):
        libsvm_file_paths = ['train.libsvm', 'train.libsvm.weights', 'libsvm_files']

        for file_path in libsvm_file_paths:
            with self.subTest(file_path=file_path):
                libsvm_path = os.path.join(self.data_path, 'libsvm', file_path)
                reader = data_utils.get_libsvm_dmatrix
                self._check_dmatrix(reader, libsvm_path, 5, 5)

    def test_parse_parquet_dmatrix(self):
        pq_file_paths = ['train.parquet', 'pq_files']

        for file_path in pq_file_paths:
            with self.subTest(file_path=file_path):
                pq_path = os.path.join(self.data_path, 'parquet', file_path)
                reader = data_utils.get_parquet_dmatrix
                self._check_dmatrix(reader, pq_path, 5, 5)

    def test_parse_parquet_dmatrix_pipe(self):
        pq_file_paths = ['pq_files']

        for file_path in pq_file_paths:
            with self.subTest(file_path=file_path):
                pq_path = os.path.join(self.data_path, 'parquet', file_path)
                pipe_dir = os.path.join(self.data_path, 'parquet', 'pipe_path')
                reader = data_utils.get_parquet_dmatrix
                is_pipe = True
                self._check_piped_dmatrix(pq_path, pipe_dir, reader, 5, 5, is_pipe)

    def test_parse_protobuf_dmatrix(self):
        pb_file_paths = ['train.pb', 'pb_files']

        for file_path in pb_file_paths:
            with self.subTest(file_path=file_path):
                pb_path = os.path.join(self.data_path, 'recordio_protobuf', file_path)
                reader = data_utils.get_recordio_protobuf_dmatrix
                self._check_dmatrix(reader, pb_path, 5, 5)

    def test_parse_protobuf_dmatrix_pipe(self):
        pb_file_paths = ['pb_files']

        for file_path in pb_file_paths:
            with self.subTest(file_path=file_path):
                pb_path = os.path.join(self.data_path, 'recordio_protobuf', file_path)
                pipe_dir = os.path.join(self.data_path, 'recordio_protobuf', 'pipe_path')
                reader = data_utils.get_recordio_protobuf_dmatrix
                is_pipe = True
                self._check_piped_dmatrix(pb_path, pipe_dir, reader, 5, 5, is_pipe)

    def test_parse_sparse_protobuf_dmatrix(self):
        pb_file_paths = ['sparse', 'sparse_edge_cases']
        dimensions = [(5, 5), (3, 25)]

        for file_path, dims in zip(pb_file_paths, dimensions):
            with self.subTest(file_path=file_path):
                pb_path = os.path.join(self.data_path, 'recordio_protobuf', file_path)
                reader = data_utils.get_recordio_protobuf_dmatrix
                self._check_dmatrix(reader, pb_path, dims[0], dims[1])

    def test_parse_protobuf_dmatrix_single_feature_label(self):
        pb_file_paths = ['single_feature_label.pb']

        for file_path in pb_file_paths:
            with self.subTest(file_path=file_path):
                pb_path = os.path.join(self.data_path, 'recordio_protobuf', file_path)
                reader = data_utils.get_recordio_protobuf_dmatrix
                self._check_dmatrix(reader, pb_path, 1, 1)
