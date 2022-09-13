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

import os
from test.utils import local_mode, test_utils

path = os.path.dirname(os.path.realpath(__file__))
boston_path = os.path.join(path, "..", "..", "resources", "boston")
data_dir = os.path.join(boston_path, "data")


def test_xgboost_boston_single_machine(docker_image, opt_ml):

    customer_script = "single_machine_customer_script.py"
    hyperparameters = {
        "objective": "reg:squarederror",
        "colsample-bytree": 0.3,
        "learning-rate": 0.1,
        "max-depth": 5,
        "reg-alpha": 10,
        "n-estimators": 10,
    }

    local_mode.train(
        customer_script, data_dir, docker_image, opt_ml, hyperparameters=hyperparameters, source_dir=boston_path
    )

    files = ["model/xgb-boston.model", "output/data/cv_results.csv", "output/data/feature-importance-plot.png"]

    assert not local_mode.file_exists(opt_ml, "output/failure"), "Failure happened"

    test_utils.files_exist(opt_ml, files)
