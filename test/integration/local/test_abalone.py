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

import contextlib
import io
import json
import os
import shutil
import tempfile
import textwrap

from test.utils import local_mode, test_utils


LIBSVM_SAMPLE = "10 1:2 2:0.645 3:0.515 4:0.15 5:1.212 6:0.515 7:0.2055 8:0.385"

MME_MODELS_URL = "http://localhost:8080/models"
MME_INVOKE_URL = "http://localhost:8080/models/{}/invoke"

path = os.path.dirname(os.path.realpath(__file__))
abalone_path = os.path.join(path, "..", "..", "resources", "abalone")
data_dir = os.path.join(abalone_path, "data")
models_dir = os.path.join(abalone_path, "models")
libsvm_model_dir = os.path.join(models_dir, "libsvm_pickled")


def get_abalone_default_hyperparameters(num_round=50):
    hyperparameters = {
        "max_depth": "5",
        "eta": "0.2",
        "gamma": "4",
        "min_child_weight": "6",
        "subsample": "0.7",
        "verbose": "1",
        "objective": "reg:linear",
        "num_round": str(num_round),
    }
    return hyperparameters


def get_libsvm_request_body(request_body=LIBSVM_SAMPLE):
    libsvm_buffer = io.BytesIO()
    libsvm_buffer.write(bytes(request_body, encoding="utf-8"))
    return libsvm_buffer.getvalue()


@contextlib.contextmanager
def append_transform_fn_to_abalone_script(abalone_path, customer_script):
    transform_fn = textwrap.dedent(
        """
        import json
        import sagemaker_xgboost_container.encoder as xgb_encoders

        def transform_fn(model, request_body, content_type, accept_type):
            dmatrix = xgb_encoders.libsvm_to_dmatrix(request_body)
            feature_contribs = model.predict(dmatrix, pred_contribs=True)
            return json.dumps(feature_contribs.tolist())
        """
    )

    tmpdir = tempfile.mkdtemp()
    with open(os.path.join(abalone_path, customer_script), "r") as fin:
        customer_script_contents = fin.read()
    with open(os.path.join(tmpdir, customer_script), "w") as fout:
        fout.write(customer_script_contents + "\n" + transform_fn)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir)


def test_xgboost_abalone_training_single_machine(docker_image, opt_ml):
    customer_script = "abalone_distributed.py"
    hyperparameters = get_abalone_default_hyperparameters()

    local_mode.train(
        customer_script,
        data_dir,
        docker_image,
        opt_ml,
        hyperparameters=hyperparameters,
        source_dir=abalone_path,
    )

    files = ["model/xgboost-model"]
    assert not local_mode.file_exists(opt_ml, "output/failure"), "Failure happened"
    test_utils.files_exist(opt_ml, files)


def test_xgboost_abalone_inference(docker_image, opt_ml):
    customer_script = "abalone_distributed.py"
    request_body = get_libsvm_request_body()

    with local_mode.serve(
        customer_script, libsvm_model_dir, docker_image, opt_ml, source_dir=abalone_path
    ):
        response_status_code, response_body = local_mode.request(
            request_body, content_type="text/libsvm"
        )

    assert response_status_code == 200
    assert not local_mode.file_exists(opt_ml, "output/failure"), "Failure happened"
    assert len(response_body.split(",")) == 1


def test_xgboost_abalone_algorithm_mode_inference(docker_image, opt_ml):
    request_body = get_libsvm_request_body()

    with local_mode.serve(
        None, libsvm_model_dir, docker_image, opt_ml, source_dir=abalone_path
    ):
        response_status_code, response_body = local_mode.request(
            request_body, content_type="text/libsvm", accept_type="application/json"
        )

    assert response_status_code == 200
    assert not local_mode.file_exists(opt_ml, "output/failure"), "Failure happened"
    assert len(response_body.split(",")) == 1
    assert '[' in response_body


def test_xgboost_abalone_custom_inference_with_transform_fn(docker_image, opt_ml):
    customer_script = "abalone_distributed.py"
    request_body = get_libsvm_request_body()
    with append_transform_fn_to_abalone_script(
        abalone_path, customer_script
    ) as custom_script_path:
        with local_mode.serve(
            customer_script,
            libsvm_model_dir,
            docker_image,
            opt_ml,
            source_dir=custom_script_path,
        ):
            response_status_code, response_body = local_mode.request(
                request_body, content_type="text/libsvm"
            )
    assert response_status_code == 200
    assert not local_mode.file_exists(opt_ml, "output/failure"), "Failure happened"
    assert (
        len(response_body.split(","))
        == len(request_body.split()) + 1  # final column is the bias term
    )


def test_xgboost_abalone_mme(docker_image, opt_ml):
    customer_script = "abalone_distributed.py"
    request_body = get_libsvm_request_body()
    additional_env_vars = [
        "SAGEMAKER_BIND_TO_PORT=8080",
        "SAGEMAKER_SAFE_PORT_RANGE=9000-9999",
        "SAGEMAKER_MULTI_MODEL=true",
    ]
    model_name = "libsvm_pickled"
    model_data = json.dumps(
        {"model_name": model_name, "url": "/opt/ml/model/{}".format(model_name)}
    )
    with local_mode.serve(
        customer_script,
        models_dir,
        docker_image,
        opt_ml,
        source_dir=abalone_path,
        additional_env_vars=additional_env_vars,
    ):
        load_status_code, _ = local_mode.request(
            model_data,
            content_type="application/json",
            request_url=MME_MODELS_URL.format(model_name),
        )
        assert load_status_code == 200
        invoke_status_code, invoke_response_body = local_mode.request(
            request_body,
            content_type="text/libsvm",
            request_url=MME_INVOKE_URL.format(model_name),
        )

    assert invoke_status_code == 200
    assert len(invoke_response_body.split(",")) == 1
    assert not local_mode.file_exists(opt_ml, "output/failure"), "Failure happened"


def test_xgboost_abalone_mme_with_transform_fn(docker_image, opt_ml):
    customer_script = "abalone_distributed.py"
    request_body = get_libsvm_request_body()
    additional_env_vars = [
        "SAGEMAKER_BIND_TO_PORT=8080",
        "SAGEMAKER_SAFE_PORT_RANGE=9000-9999",
        "SAGEMAKER_MULTI_MODEL=true",
    ]
    model_name = "libsvm_pickled"
    model_data = json.dumps(
        {"model_name": model_name, "url": "/opt/ml/model/{}".format(model_name)}
    )
    with append_transform_fn_to_abalone_script(
        abalone_path, customer_script
    ) as custom_script_path:
        with local_mode.serve(
            customer_script,
            models_dir,
            docker_image,
            opt_ml,
            source_dir=custom_script_path,
            additional_env_vars=additional_env_vars,
        ):
            load_status_code, _ = local_mode.request(
                model_data,
                content_type="application/json",
                request_url=MME_MODELS_URL.format(model_name),
            )
            assert load_status_code == 200
            invoke_status_code, invoke_response_body = local_mode.request(
                request_body,
                content_type="text/libsvm",
                request_url=MME_INVOKE_URL.format(model_name),
            )

    assert invoke_status_code == 200
    assert (
        len(invoke_response_body.split(","))
        == len(request_body.split()) + 1  # final column is the bias term
    )
    assert not local_mode.file_exists(opt_ml, "output/failure"), "Failure happened"
