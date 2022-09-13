# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from test.utils import local_mode, test_utils

import pytest

path = os.path.dirname(os.path.realpath(__file__))
data_root = os.path.join(path, "..", "..", "resources")


def get_abalone_default_hyperparameters(num_round=50):
    hyperparameters = {
        "max_depth": "5",
        "eta": "0.2",
        "gamma": "4",
        "min_child_weight": "6",
        "subsample": "0.7",
        "num_round": str(num_round),
    }
    return hyperparameters


@pytest.mark.parametrize(
    "dataset,extra_hps,model_file_count",
    [
        ("abalone", {"objective": "reg:squarederror", "_kfold": "5"}, 5),
        ("abalone-binary", {"objective": "binary:logistic", "_kfold": "5"}, 5),
        ("abalone-multiclass", {"objective": "multi:softprob", "num_class": "4", "_kfold": "5"}, 5),
        ("abalone", {"objective": "reg:squarederror", "_kfold": "5", "_num_cv_round": "2"}, 10),
        (
            "abalone-multiclass",
            {"objective": "multi:softprob", "num_class": "4", "_kfold": "5", "_num_cv_round": "3"},
            15,
        ),
    ],
)
def test_xgboost_abalone_kfold(dataset, extra_hps, model_file_count, docker_image, opt_ml):
    hyperparameters = get_abalone_default_hyperparameters()
    data_path = os.path.join(data_root, dataset, "data")

    local_mode.train(
        False,
        data_path,
        docker_image,
        opt_ml,
        hyperparameters={**hyperparameters, **extra_hps},
    )

    files = [f"model/xgboost-model-{i}" for i in range(model_file_count)]
    assert not local_mode.file_exists(opt_ml, "output/failure"), "Failure happened"
    test_utils.files_exist(opt_ml, files)
    local_mode.file_exists(opt_ml, "output/data/predictions.csv")
