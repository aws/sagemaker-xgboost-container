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
from __future__ import absolute_import

import os
from test.utils import local_mode

path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.join(path, "..", "..", "resources", "versions")
abalone_path = os.path.join(path, "..", "..", "resources", "abalone")
data_dir = os.path.join(abalone_path, "data")


def test_package_version(docker_image, opt_ml):
    version_check_script = "train.py"

    local_mode.train(
        version_check_script,
        data_dir,
        docker_image,
        opt_ml,
        source_dir=script_path,
    )

    assert not local_mode.file_exists(opt_ml, "output/failure"), "Failure happened"
