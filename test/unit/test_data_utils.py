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
import subprocess
import time

from sagemaker_algorithm_toolkit import exceptions as exc
from sagemaker_xgboost_container import data_utils


class TestTrainUtils(unittest.TestCase):

    def setUp(self):
        path = os.path.abspath(__file__)
        self.resource_path = os.path.join(os.path.dirname(path), '..', 'resources')
        self.utils_path = os.path.join(os.path.dirname(path), '..', 'utils')

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
                csv_path = os.path.join(self.resource_path, 'csv', file_path)
                data_utils.validate_data_file_path(csv_path, 'csv')

    def test_validate_libsvm_files(self):
        libsvm_file_paths = ['train.libsvm', 'train.libsvm.weights', 'libsvm_files']

        for file_path in libsvm_file_paths:
            with self.subTest(file_path=file_path):
                csv_path = os.path.join(self.resource_path, 'libsvm', file_path)
                data_utils.validate_data_file_path(csv_path, 'libsvm')

    def test_parse_csv_dmatrix(self):
        csv_file_paths_and_weight = [('train.csv', 0), ('train.csv.weights', 1), ('csv_files', 0)]

        for file_path, csv_weight in csv_file_paths_and_weight:
            with self.subTest(file_path=file_path, csv_weight=csv_weight):
                csv_path = os.path.join(self.resource_path, 'csv', file_path)

                single_node_dmatrix = data_utils.get_csv_dmatrix(csv_path, csv_weight)

                self.assertEqual(5, single_node_dmatrix.num_col())
                self.assertEqual(5, single_node_dmatrix.num_row())

                no_weight_test_features = ["f{}".format(idx) for idx in range(single_node_dmatrix.num_col())]

                self.assertEqual(no_weight_test_features, single_node_dmatrix.feature_names)

    def test_parse_csv_dmatrix_pipe(self):
        csv_file_paths_and_weight = [('csv_files', 0)]

        for file_path, csv_weight in csv_file_paths_and_weight:
            with self.subTest(file_path=file_path, csv_weight=csv_weight):
                csv_path = os.path.join(self.resource_path, 'csv', file_path)
                pipe_dir = os.path.join(self.resource_path, 'csv/pipe_path')
                pipe_path = os.path.join(pipe_dir, 'train')
                os.system('python3 {}/sagemaker_pipe.py train {} {}&'.format(self.utils_path,
                                                                             csv_path,
                                                                             pipe_dir))

                time.sleep(1)

                single_node_dmatrix = data_utils.get_csv_dmatrix_pipe_mode(pipe_path, csv_weight)

                self.assertEqual(5, single_node_dmatrix.num_col())
                self.assertEqual(5, single_node_dmatrix.num_row())

                no_weight_test_features = ["f{}".format(idx) for idx in range(single_node_dmatrix.num_col())]

                self.assertEqual(no_weight_test_features, single_node_dmatrix.feature_names)

                pids = subprocess.check_output(['pidof', 'python3'])
                pids = pids.decode('utf-8').split(' ')
                os.system('kill {}'.format(pids[0]))
                os.system('rm {}*'.format(pipe_path))

    def test_parse_libsvm_dmatrix(self):
        libsvm_file_paths = ['train.libsvm', 'train.libsvm.weights', 'libsvm_files']

        for file_path in libsvm_file_paths:
            with self.subTest(file_path=file_path):
                libsvm_path = os.path.join(self.resource_path, 'libsvm', file_path)

                single_node_dmatrix = data_utils.get_libsvm_dmatrix(libsvm_path)

                self.assertEqual(5, single_node_dmatrix.num_col())
                self.assertEqual(5, single_node_dmatrix.num_row())

                no_weight_test_features = ["f{}".format(idx) for idx in range(single_node_dmatrix.num_col())]

                self.assertEqual(no_weight_test_features, single_node_dmatrix.feature_names)

    def test_parse_parquet_dmatrix(self):
        pq_file_paths = ['train.parquet', 'pq_files']

        for file_path in pq_file_paths:
            with self.subTest(file_path=file_path):
                pq_path = os.path.join(self.resource_path, 'parquet', file_path)

                single_node_dmatrix = data_utils.get_parquet_dmatrix(pq_path)

                self.assertEqual(5, single_node_dmatrix.num_col())
                self.assertEqual(5, single_node_dmatrix.num_row())

                no_weight_test_features = ["f{}".format(idx) for idx in range(single_node_dmatrix.num_col())]

                self.assertEqual(no_weight_test_features, single_node_dmatrix.feature_names)

    def test_parse_parquet_dmatrix_pipe(self):
        pq_file_paths = ['pq_files']

        for file_path in pq_file_paths:
            with self.subTest(file_path=file_path):
                pq_path = os.path.join(self.resource_path, 'parquet', file_path)
                pipe_dir = os.path.join(self.resource_path, 'parquet/pipe_path')
                pipe_path = os.path.join(pipe_dir, 'train')
                os.system('python3 {}/sagemaker_pipe.py train {} {}&'.format(self.utils_path,
                                                                             pq_path,
                                                                             pipe_dir))

                time.sleep(1)

                single_node_dmatrix = data_utils.get_parquet_dmatrix_pipe_mode(pipe_path)

                self.assertEqual(5, single_node_dmatrix.num_col())
                self.assertEqual(5, single_node_dmatrix.num_row())

                no_weight_test_features = ["f{}".format(idx) for idx in range(single_node_dmatrix.num_col())]

                self.assertEqual(no_weight_test_features, single_node_dmatrix.feature_names)

                pids = subprocess.check_output(['pidof', 'python3'])
                pids = pids.decode('utf-8').split(' ')
                os.system('kill {}'.format(pids[0]))
                os.system('rm {}*'.format(pipe_path))

    def test_parse_protobuf_dmatrix(self):
        pb_file_paths = ['train.pb', 'pb_files']

        for file_path in pb_file_paths:
            with self.subTest(file_path=file_path):
                pb_path = os.path.join(self.resource_path, 'recordio_protobuf', file_path)

                single_node_dmatrix = data_utils.get_recordio_protobuf_dmatrix(pb_path)

                self.assertEqual(5, single_node_dmatrix.num_col())
                self.assertEqual(5, single_node_dmatrix.num_row())

                no_weight_test_features = ["f{}".format(idx) for idx in range(single_node_dmatrix.num_col())]

                self.assertEqual(no_weight_test_features, single_node_dmatrix.feature_names)

    def test_parse_protobuf_dmatrix_pipe(self):
        pb_file_paths = ['pb_files']

        for file_path in pb_file_paths:
            with self.subTest(file_path=file_path):
                pb_path = os.path.join(self.resource_path, 'recordio_protobuf', file_path)
                pipe_dir = os.path.join(self.resource_path, 'recordio_protobuf/pipe_path')
                pipe_path = os.path.join(pipe_dir, 'train')
                os.system('python3 {}/sagemaker_pipe.py train {} {}&'.format(self.utils_path,
                                                                             pb_path,
                                                                             pipe_dir))

                time.sleep(1)

                single_node_dmatrix = data_utils.get_recordio_protobuf_dmatrix_pipe_mode(pipe_path)

                self.assertEqual(5, single_node_dmatrix.num_col())
                self.assertEqual(5, single_node_dmatrix.num_row())

                no_weight_test_features = ["f{}".format(idx) for idx in range(single_node_dmatrix.num_col())]

                self.assertEqual(no_weight_test_features, single_node_dmatrix.feature_names)

                pids = subprocess.check_output(['pidof', 'python3'])
                pids = pids.decode('utf-8').split(' ')
                os.system('kill {}'.format(pids[0]))
                os.system('rm {}*'.format(pipe_path))
