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
import re
import shutil
import tempfile
import platform

from test.utils import local_mode, test_utils


# use dataset from https://github.com/dmlc/xgboost/tree/master/demo/data
path = os.path.dirname(os.path.realpath(__file__))
agaricus_path = os.path.join(path, '..', '..', 'resources', 'agaricus')
data_dir = os.path.join(agaricus_path, 'data')
checkpoint_dir = os.path.join(agaricus_path, "checkpoints")

iteration_regex = r"\[([0-9]+)\].*train-rmse:"
train_rmse_regex = r"\[[0-9]+\].*train-rmse:([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?).*"

script = """
    from sagemaker_xgboost_container.training import main

    if __name__ == "__main__":
        main()
    """


def test_xgboost_agaricus_checkpoint(docker_image, opt_ml, capfd):

    hyperparameters = {"num_round": 10}

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write(script)

    with tempfile.TemporaryDirectory() as temp_dir:

        # Docker cannot mount Mac OS /var folder properly see
        # https://forums.docker.com/t/var-folders-isnt-mounted-properly/9600
        mount_dir = "/private" + temp_dir if platform.system() == "Darwin" else temp_dir

        additional_volumes = ["{}:/opt/ml/checkpoints".format(mount_dir)]

        local_mode.train(
            temp_file.name, data_dir, docker_image, opt_ml,
            hyperparameters=hyperparameters,
            content_type="libsvm",
            additional_volumes=additional_volumes,
            script_mode=False)

        files = os.listdir(mount_dir)
        assert len(files) == 5  # default is to save the 5 latest checkpoints
        assert all(f.startswith("xgboost-checkpoint.") for f in files)
        assert all(0 <= int(f.split('.')[1]) < 10 for f in files)

    os.remove(temp_file.name)

    assert not local_mode.file_exists(opt_ml, 'output/failure')

    captured = capfd.readouterr()
    iterations = [int(i) for i in re.findall(iteration_regex, captured.out)]
    assert iterations == list(range(10))


def test_xgboost_agaricus_resume(docker_image, opt_ml, capfd):

    hyperparameters = {"num_round": 20}

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write(script)

    with tempfile.TemporaryDirectory() as temp_dir:

        mount_dir = "/private" + temp_dir if platform.system() == "Darwin" else temp_dir

        for checkpoint in os.listdir(checkpoint_dir):
            shutil.copyfile(
                os.path.join(checkpoint_dir, checkpoint),
                os.path.join(mount_dir, checkpoint))

        additional_volumes = ["{}:/opt/ml/checkpoints".format(mount_dir)]

        local_mode.train(
            temp_file.name, data_dir, docker_image, opt_ml,
            hyperparameters=hyperparameters,
            content_type="libsvm",
            additional_volumes=additional_volumes,
            script_mode=False)

        files = os.listdir(mount_dir)
        assert len(files) == 10  # previous 5 + latest 5, previous checkpoints are not deleted
        assert all(f.startswith("xgboost-checkpoint.") for f in files)
        assert all(5 <= int(f.split('.')[1]) < 10 or 15 <= int(f.split('.')[1]) < 20 for f in files)

    os.remove(temp_file.name)

    assert not local_mode.file_exists(opt_ml, 'output/failure')

    # resume_agaricus.py should load checkpoint from round 9 and train from round 10 to 19
    captured = capfd.readouterr()
    iterations = [int(i) for i in re.findall(iteration_regex, captured.out)]
    assert iterations == list(range(10, 20))

    # training from scratch, train-rmse on agaricus.txt decreases
    # from 0.35 at round 0 to about 0.019 after round 9
    train_rmse = [float(i) for i in re.findall(train_rmse_regex, captured.out)]
    assert len(iterations) == len(train_rmse)
    assert all(x < 0.02 for x in train_rmse)
