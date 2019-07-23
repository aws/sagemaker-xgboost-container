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

from sagemaker_xgboost_container import data_utils


class TestTrainUtils(unittest.TestCase):

    def setUp(self):
        path = os.path.abspath(__file__)
        self.resource_path = os.path.join(os.path.dirname(path), 'resources')

    def test_get_libsvm_dmatrix_no_weights(self):
        libsvm_path = os.path.join(self.resource_path, 'libsvm')

        single_node_dmatrix = data_utils.get_libsvm_dmatrix(libsvm_path)

        self.assertEqual(5, single_node_dmatrix.num_col())
        self.assertEqual(40, single_node_dmatrix.num_row())

        no_weight_test_features = ["f{}".format(idx) for idx in range(single_node_dmatrix.num_col())]

        self.assertEqual(no_weight_test_features, single_node_dmatrix.feature_names)

    def test_get_libsvm_dmatrix_weights(self):
        libsvm_path = os.path.join(self.resource_path, 'libsvm-weights')

        single_node_dmatrix = data_utils.get_libsvm_dmatrix(libsvm_path)

        self.assertEqual(5, single_node_dmatrix.num_col())
        self.assertEqual(40, single_node_dmatrix.num_row())

        no_weight_test_features = ["f{}".format(idx) for idx in range(single_node_dmatrix.num_col())]

        self.assertEqual(no_weight_test_features, single_node_dmatrix.feature_names)

    def test_get_csv_dmatrix_no_weights(self):
        csv_path = os.path.join(self.resource_path, 'csv')

        single_node_dmatrix = data_utils.get_csv_dmatrix(csv_path, 0)

        self.assertEqual(5, single_node_dmatrix.num_col())
        self.assertEqual(40, single_node_dmatrix.num_row())

        no_weight_test_features = ["f{}".format(idx) for idx in range(single_node_dmatrix.num_col())]

        self.assertEqual(no_weight_test_features, single_node_dmatrix.feature_names)

    def test_get_csv_dmatrix_weights(self):
        csv_path = os.path.join(self.resource_path, 'csv-weights')

        single_node_dmatrix = data_utils.get_csv_dmatrix(csv_path, 1)

        self.assertEqual(5, single_node_dmatrix.num_col())
        self.assertEqual(40, single_node_dmatrix.num_row())

        no_weight_test_features = ["f{}".format(idx) for idx in range(single_node_dmatrix.num_col())]

        self.assertEqual(no_weight_test_features, single_node_dmatrix.feature_names)
