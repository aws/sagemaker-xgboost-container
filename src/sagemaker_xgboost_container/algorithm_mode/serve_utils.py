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
import io
import json
import numpy as np
import os
import pickle as pkl

from scipy.sparse import csr_matrix
from sagemaker_containers.record_pb2 import Record
from sagemaker_containers._recordio import _write_recordio
import xgboost as xgb

from sagemaker_xgboost_container import encoder
from sagemaker_xgboost_container.algorithm_mode import integration
from sagemaker_xgboost_container.constants.sm_env_constants import SAGEMAKER_INFERENCE_OUTPUT, SAGEMAKER_BATCH
from sagemaker_xgboost_container.data_utils import CSV, LIBSVM, RECORDIO_PROTOBUF, get_content_type
from sagemaker_xgboost_container.constants.xgb_constants import (BINARY_HINGE, BINARY_LOG, BINARY_LOGRAW,
                                                                 MULTI_SOFTMAX, MULTI_SOFTPROB, REG_GAMMA,
                                                                 REG_LOG, REG_SQUAREDERR, REG_TWEEDIE)
from sagemaker_xgboost_container.encoder import json_to_jsonlines


logging = integration.setup_main_logger(__name__)

PKL_FORMAT = 'pkl_format'
XGB_FORMAT = 'xgb_format'

# classification selectable inference keys
PREDICTED_LABEL = "predicted_label"
LABELS = "labels"
PROBABILITY = "probability"
PROBABILITIES = "probabilities"
RAW_SCORE = "raw_score"
RAW_SCORES = "raw_scores"

# regression selectable inference keys
PREDICTED_SCORE = "predicted_score"

# all supported selecable content keys
ALL_VALID_SELECT_KEYS = [PREDICTED_LABEL, LABELS, PROBABILITY, PROBABILITIES, RAW_SCORE, RAW_SCORES, PREDICTED_SCORE]

# mapping of xgboost objective functions to valid selectable inference content
VALID_OBJECTIVES = {
    REG_SQUAREDERR: [PREDICTED_SCORE],
    REG_LOG: [PREDICTED_SCORE],
    REG_GAMMA: [PREDICTED_SCORE],
    REG_TWEEDIE: [PREDICTED_SCORE],
    BINARY_LOG: [PREDICTED_LABEL, LABELS, PROBABILITY, PROBABILITIES, RAW_SCORE, RAW_SCORES],
    BINARY_LOGRAW: [PREDICTED_LABEL, LABELS, RAW_SCORE, RAW_SCORES],
    BINARY_HINGE: [PREDICTED_LABEL, LABELS, RAW_SCORE, RAW_SCORES],
    MULTI_SOFTMAX: [PREDICTED_LABEL, LABELS, RAW_SCORE, RAW_SCORES],
    MULTI_SOFTPROB: [PREDICTED_LABEL, LABELS, PROBABILITY, PROBABILITIES, RAW_SCORE, RAW_SCORES]
}


# FIXME: https://github.com/aws/sagemaker-xgboost-container/issues/12
# This was failing a previous container test; need to make decision as to change test or behavior.
def _get_sparse_matrix_from_libsvm(payload):
    pylist = map(lambda x: x.split(' '), payload.split('\n'))
    colon = ':'
    row = []
    col = []
    data = []
    for row_idx, line in enumerate(pylist):
        for item in line:
            if colon in item:
                col_idx = item.split(colon)[0]
                val = item.split(colon)[1]
                row.append(row_idx)
                col.append(col_idx)
                data.append(val)

    row = np.array(row)
    col = np.array(col).astype(np.int)
    data = np.array(data).astype(np.float)
    if not (len(row) == len(col) and len(col) == len(data)):
        raise RuntimeError("Dimension checking failed when transforming sparse matrix.")

    return csr_matrix((data, (row, col)))


def parse_content_data(input_data, input_content_type):
    dtest = None
    content_type = get_content_type(input_content_type)
    payload = input_data
    if content_type == CSV:
        try:
            decoded_payload = payload.strip().decode("utf-8")
            dtest = encoder.csv_to_dmatrix(decoded_payload, dtype=np.float)
        except Exception as e:
            raise RuntimeError("Loading csv data failed with Exception, "
                               "please ensure data is in csv format:\n {}\n {}".format(type(e), e))
    elif content_type == LIBSVM:
        try:
            decoded_payload = payload.strip().decode("utf-8")
            dtest = xgb.DMatrix(_get_sparse_matrix_from_libsvm(decoded_payload))
        except Exception as e:
            raise RuntimeError("Loading libsvm data failed with Exception, "
                               "please ensure data is in libsvm format:\n {}\n {}".format(type(e), e))
    elif content_type == RECORDIO_PROTOBUF:
        try:
            dtest = encoder.recordio_protobuf_to_dmatrix(payload)
        except Exception as e:
            raise RuntimeError("Loading recordio-protobuf data failed with "
                               "Exception, please ensure data is in "
                               "recordio-protobuf format: {} {}".format(type(e), e))
    else:
        raise RuntimeError("Content-type {} is not supported.".format(input_content_type))

    return dtest, content_type


def get_loaded_booster(model_dir):
    model_files = (data_file for data_file in os.listdir(model_dir)
                   if os.path.isfile(os.path.join(model_dir, data_file)))
    model_file = next(model_files)
    try:
        booster = pkl.load(open(os.path.join(model_dir, model_file), 'rb'))
        format = PKL_FORMAT
    except Exception as exp_pkl:
        try:
            booster = xgb.Booster()
            booster.load_model(os.path.join(model_dir, model_file))
            format = XGB_FORMAT
        except Exception as exp_xgb:
            raise RuntimeError("Model at {} cannot be loaded:\n{}\n{}".format(model_dir, str(exp_pkl), str(exp_xgb)))
    booster.set_param('nthread', 1)
    return booster, format


def predict(booster, model_format, dtest, input_content_type):
    if model_format == PKL_FORMAT:
        x = len(booster.feature_names)
        y = len(dtest.feature_names)

        try:
            content_type = get_content_type(input_content_type)
        except Exception:
            raise ValueError('Content type {} is not supported'.format(input_content_type))

        if content_type == LIBSVM:
            if y > x + 1:
                raise ValueError('Feature size of libsvm inference data {} is larger than '
                                 'feature size of trained model {}.'.format(y, x))
        elif content_type in [CSV, RECORDIO_PROTOBUF]:
            if not ((x == y) or (x == y + 1)):
                raise ValueError('Feature size of {} inference data {} is not consistent '
                                 'with feature size of trained model {}.'.
                                 format(content_type, y, x))
        else:
            raise ValueError('Content type {} is not supported'.format(content_type))
    return booster.predict(dtest,
                           ntree_limit=getattr(booster, "best_ntree_limit", 0),
                           validate_features=False)


def is_selectable_inference_response():
    return SAGEMAKER_INFERENCE_OUTPUT in os.environ


def get_selected_content_keys():
    """Get the selected content keys from the `SAGEMAKER_INFERENCE_OUTPUT` env var.

    :return: selected content keys (list of str)
    """
    if is_selectable_inference_response():
        return os.getenv(SAGEMAKER_INFERENCE_OUTPUT).replace(' ', '').lower().split(',')
    raise RuntimeError("'SAGEMAKER_INFERENCE_OUTPUT' environment variable is not present. "
                       "Selectable inference content is not enabled.")


def _get_labels(objective, num_class=""):
    if "binary:" in objective:
        return [0, 1]
    if "multi:" in objective and num_class:
        return list(range(int(num_class)))
    return np.nan


def _get_predicted_label(objective, data):
    if objective in [BINARY_HINGE, MULTI_SOFTMAX]:
        return data.item()
    if objective in [BINARY_LOG]:
        return int(data > 0.5)
    if objective in [BINARY_LOGRAW]:
        return int(data > 0)
    if objective in [MULTI_SOFTPROB]:
        return np.argmax(data).item()
    return np.nan


def _get_probability(objective, data):
    if objective in [MULTI_SOFTPROB]:
        return max(data).item()
    if objective in [BINARY_LOG]:
        return data.item()
    return np.nan


def _get_probabilities(objective, data):
    if objective in [MULTI_SOFTPROB]:
        return data.tolist()
    if objective in [BINARY_LOG]:
        classone_probs = data.item()
        classzero_probs = 1.0 - classone_probs
        return [classzero_probs, classone_probs]
    return np.nan


def _get_raw_score(objective, data):
    if objective in [MULTI_SOFTPROB]:
        return max(data).item()
    if objective in [BINARY_LOGRAW, BINARY_HINGE, BINARY_LOG, MULTI_SOFTMAX]:
        return data.item()
    return np.nan


def _get_raw_scores(objective, data):
    if objective in [MULTI_SOFTPROB]:
        return data.tolist()
    if objective in [BINARY_LOGRAW, BINARY_HINGE, BINARY_LOG, MULTI_SOFTMAX]:
        classone_probs = data.item()
        classzero_probs = 1.0 - classone_probs
        return [classzero_probs, classone_probs]
    return np.nan


def get_selected_content(data, keys, objective, num_class=""):
    """Build the selected content dictionary based on the objective function and requested content.

    :param data: output of xgboost content (list of numpy objects)
    :param keys: strings denoting selected keys (list of str)
    :param objective: objective xgboost training function (str)
    :param num_class: number of classes for multiclass classification (str, optional)
    :return: selected content (list of dict)
    """
    if objective not in VALID_OBJECTIVES:
        raise ValueError("Objective `{}` unsupported for selectable inference content.".format(objective))

    valid_selected_keys = set(keys).intersection(VALID_OBJECTIVES[objective])
    invalid_selected_keys = set(keys).difference(VALID_OBJECTIVES[objective])
    if invalid_selected_keys:
        logging.warning("Selected key(s) {} incompatible for objective '{}'. "
                        "Please use list of compatible selectable inference content: {}"
                        .format(invalid_selected_keys, objective, VALID_OBJECTIVES[objective]))

    content = []
    for prediction in data:
        output = {}
        if PREDICTED_LABEL in valid_selected_keys:
            output[PREDICTED_LABEL] = _get_predicted_label(objective, prediction)
        if LABELS in valid_selected_keys:
            output[LABELS] = _get_labels(objective, num_class=num_class)
        if PROBABILITY in valid_selected_keys:
            output[PROBABILITY] = _get_probability(objective, prediction)
        if PROBABILITIES in valid_selected_keys:
            output[PROBABILITIES] = _get_probabilities(objective, prediction)
        if RAW_SCORE in valid_selected_keys:
            output[RAW_SCORE] = _get_raw_score(objective, prediction)
        if RAW_SCORES in valid_selected_keys:
            output[RAW_SCORES] = _get_raw_scores(objective, prediction)
        if PREDICTED_SCORE in valid_selected_keys:
            output[PREDICTED_SCORE] = prediction.item()
        if invalid_selected_keys:
            for invalid_selected_key in invalid_selected_keys:
                output[invalid_selected_key] = np.nan
        content.append(output)
    return content


def _encode_selected_content_csv(content, ordered_keys_list):
    """Encode content is csv format with the ordered_keys_list as the header.

    :param content: list of selected content
    :param ordered_keys_list: list of selected content keys
    :return: csv string
    """
    def _generate_single_csv_line_selected_content(content, ordered_keys_list):
        """Generate a single csv line response for selectable inference content

        :param content: list of selected content
        :param ordered_keys_list: list of selected content keys
        :return: a generate that produces a csv line for each datapoint
        """
        for single_prediction in content:
            values = []
            for key in ordered_keys_list:
                if isinstance(single_prediction[key], list):
                    value = '"{}"'.format(single_prediction[key])
                else:
                    value = str(single_prediction[key])
                values.append(value)
            yield ','.join(values)

    return '\n'.join(_generate_single_csv_line_selected_content(content, ordered_keys_list))


def _write_record(record, key, value):
    record.label[key].float32_tensor.values.extend(value)


def _encode_selected_content_recordio_protobuf(content):
    """Encode list of dictionaries into recordio protobuf format. Dictionary keys are "label" keys.

    :param content: list of dictionaries
    :return: recordio bytes
    """
    record_bio = io.BytesIO()
    recordio_bio = io.BytesIO()
    record = Record()
    for item in content:
        for key in item.keys():
            value = item[key] if type(item[key]) is list else [item[key]]
            _write_record(record, key, value)
        record_bio.write(record.SerializeToString())
        record.Clear()
        _write_recordio(recordio_bio, record_bio.getvalue())
    return recordio_bio.getvalue()


def encode_selected_content(content, keys, accept):
    """Encodes the selected content and keys based on the given accept type.

    :param content: list of selected content. See example below.
                    [{"predicted_label": 1, "probabilities": [0.4, 0.6]},
                     {"predicted_label": 0, "probabilities": [0.9, 0.1]}]
    :param keys: list of strings denoting selected content
    :param accept: accept mime-type
    :return: encoded content
    """
    if accept == "application/json":
        return json.dumps({"predictions": content})
    if accept == "application/jsonlines":
        return json_to_jsonlines({"predictions": content})
    if accept == "application/x-recordio-protobuf":
        return _encode_selected_content_recordio_protobuf(content)
    if accept == "text/csv":
        csv_response = _encode_selected_content_csv(content, keys)
        if os.getenv(SAGEMAKER_BATCH):
            return csv_response + '\n'
        return csv_response
    raise RuntimeError("Cannot encode selected content into accept type '{}'.".format(accept))
