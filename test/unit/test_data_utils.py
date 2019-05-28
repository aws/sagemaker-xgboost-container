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

from sagemaker_algorithm_toolkit import exceptions as exc
from sagemaker_xgboost_container import data_utils


class TestTrainUtils(unittest.TestCase):

    def setUp(self):
        path = os.path.abspath(__file__)
        self.resource_path = os.path.join(os.path.dirname(path), '..', 'resources')

    def test_get_content_type(self):
        self.assertEqual('libsvm', data_utils.get_content_type('libsvm'))
        self.assertEqual('libsvm', data_utils.get_content_type('text/libsvm'))
        self.assertEqual('libsvm', data_utils.get_content_type('text/x-libsvm'))

        self.assertEqual('csv', data_utils.get_content_type('csv'))
        self.assertEqual('csv', data_utils.get_content_type('text/csv'))

        with self.assertRaises(exc.UserError):
            data_utils.get_content_type('incorrect_format')

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
