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
from sagemaker_xgboost_container.constants import sm_env_constants
from sagemaker_xgboost_container.data_utils import CSV, LIBSVM, RECORDIO_PROTOBUF, get_content_type
from sagemaker_xgboost_container.constants.xgb_constants import (BINARY_HINGE, BINARY_LOG, BINARY_LOGRAW,
                                                                 MULTI_SOFTMAX, MULTI_SOFTPROB, REG_GAMMA,
                                                                 REG_LOG, REG_SQUAREDERR, REG_TWEEDIE)
from sagemaker_xgboost_container.encoder import json_to_jsonlines


logging = integration.setup_main_logger(__name__)

PKL_FORMAT = 'pkl_format'
XGB_FORMAT = 'xgb_format'

SAGEMAKER_BATCH = os.getenv(sm_env_constants.SAGEMAKER_BATCH)
SAGEMAKER_INFERENCE_OUTPUT = os.getenv(sm_env_constants.SAGEMAKER_INFERENCE_OUTPUT)

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
    models = []
    formats = []
    for model_file in model_files:
        path = os.path.join(model_dir, model_file)
        logging.info(f"Loading the model from {path}")
        try:
            booster = pkl.load(open(path, 'rb'))
            format = PKL_FORMAT
        except Exception as exp_pkl:
            try:
                booster = xgb.Booster()
                booster.load_model(path)
                format = XGB_FORMAT
            except Exception as exp_xgb:
                raise RuntimeError("Model at {} cannot be loaded:\n{}\n{}".format(path, str(exp_pkl), str(exp_xgb)))
        booster.set_param('nthread', 1)
        models.append(booster)
        formats.append(format)

    return models, formats


def predict(models, model_format, dtest, input_content_type):
    if model_format[0] == PKL_FORMAT:
        x = len(models[0].feature_names)
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

    ensemble = []
    for booster in models:
        preds = booster.predict(dtest,
                                ntree_limit=getattr(booster, "best_ntree_limit", 0),
                                validate_features=False)
        ensemble.append(preds)

    return ensemble[1] if len(ensemble) == 1 else np.mean(ensemble, axis=0)


def is_selectable_inference_output():
    return sm_env_constants.SAGEMAKER_INFERENCE_OUTPUT in os.environ


def get_selected_output_keys():
    """Get the selected output content keys from the `SAGEMAKER_INFERENCE_OUTPUT` env var.

    :return: selected output content keys (list of str)
    """
    if is_selectable_inference_output():
        return SAGEMAKER_INFERENCE_OUTPUT.replace(' ', '').lower().split(',')
    raise RuntimeError("'SAGEMAKER_INFERENCE_OUTPUT' environment variable is not present. "
                       "Selectable inference content is not enabled.")


def _get_labels(objective, num_class=""):
    """Get the labels for a classification problem.

    Based on the implementation in the xgboost sklearn API
    https://github.com/dmlc/xgboost/blob/release_1.0.0/python-package/xgboost/sklearn.py#L844-L902.

    :param objective: xgboost objective function (str)
    :param num_class: number of classes in multiclass (str, optional)
    :return: classes (list of integers) or np.nan
    """
    if "binary:" in objective:
        return [0, 1]
    if "multi:" in objective and num_class:
        return list(range(int(num_class)))
    return np.nan


def _get_predicted_label(objective, raw_prediction):
    """Get the predicted label for a classification problem.

    :param objective: xgboost objective function (str)
    :param raw_prediction: xgboost predict output (numpy array or numpy primitive)
    :return: predicted label (int or float) or np.nan
    """
    if objective in [BINARY_HINGE, MULTI_SOFTMAX]:
        return raw_prediction.item()
    if objective in [BINARY_LOG]:
        return int(raw_prediction > 0.5)
    if objective in [BINARY_LOGRAW]:
        return int(raw_prediction > 0)
    if objective in [MULTI_SOFTPROB]:
        return np.argmax(raw_prediction).item()
    return np.nan


def _get_probability(objective, raw_prediction):
    """Get the probability score for a classification problem.

    In binary classification, this will return the probability score of the class
    being predicted as '1.0' or '1'. In multiclass classification, this will return
    the probability score of the winning class.

    :param objective: xgboost objective function (str)
    :param raw_prediction: xgboost predict output (numpy array or numpy primitive)
    :return: probability score (float) or np.nan
    """
    if objective in [MULTI_SOFTPROB]:
        return max(raw_prediction).item()
    if objective in [BINARY_LOG]:
        return raw_prediction.item()
    return np.nan


def _get_probabilities(objective, raw_prediction):
    """Get the probability scores for all classes for a classification problem.

    :param objective: xgboost objective function (str)
    :param raw_prediction: xgboost predict output (numpy array or numpy primitive)
    :return: probability scores (list of floats) or np.nan
    """
    if objective in [MULTI_SOFTPROB]:
        return raw_prediction.tolist()
    if objective in [BINARY_LOG]:
        classone_probs = raw_prediction.item()
        classzero_probs = 1.0 - classone_probs
        return [classzero_probs, classone_probs]
    return np.nan


def _get_raw_score(objective, raw_prediction):
    """Get the raw score for a classification problem.

    A raw score is defined as any numeric value. The higher the value, the more likely
    the class is to be predicted as the "predicted_label". In binary classification,
    this represents the likelihood of a class being predicted as '1.0' or '1' In
    multiclass, it represents the likelihood of the winning class.

    Note that a 'probability' can be considered as a 'raw score', but a 'raw score' may
    not necessarily be a 'probability'.

    :param objective: xgboost objective function (str)
    :param raw_prediction: xgboost predict output (numpy array or numpy primitive)
    :return: raw score (float) or np.nan
    """
    if objective in [MULTI_SOFTPROB]:
        return max(raw_prediction).item()
    if objective in [BINARY_LOGRAW, BINARY_HINGE, BINARY_LOG, MULTI_SOFTMAX]:
        return raw_prediction.item()
    return np.nan


def _get_raw_scores(objective, raw_prediction):
    """Get the raw scores for all classes for a classification problem.

    A raw score is defined as any numeric value. The higher the value, the more likely
    the class is to be predicted as the "predicted_label". In binary classification,
    this represents the likelihood of a class being predicted as '1.0' or '1' In
    multiclass, it represents the likelihood of the winning class.

    Note that a 'probability' can be considered as a 'raw score', but a 'raw score' may
    not necessarily be a 'probability'.

    :param objective: xgboost objective function (str)
    :param raw_prediction: xgboost predict output (numpy array or numpy primitive)
    :return: raw scores (list of floats) or np.nan
    """
    if objective in [MULTI_SOFTPROB]:
        return raw_prediction.tolist()
    if objective in [BINARY_LOGRAW, BINARY_HINGE, BINARY_LOG, MULTI_SOFTMAX]:
        classone_probs = raw_prediction.item()
        classzero_probs = 1.0 - classone_probs
        return [classzero_probs, classone_probs]
    return np.nan


def get_selected_predictions(raw_predictions, selected_keys, objective, num_class=""):
    """Build the selected prediction dictionary based on the objective function and
    requested information.

    'raw_predictions' is the output of ScoringService.predict(...) and will change
    depending on the objective xgboost training function used. For each prediction,
    a new dictionary will be built with the selected content requested in 'selected_keys'.

    VALID_OBJECTIVES contains a mapping of objective functions to valid selected keys.
    For example, a booster trained with a "reg:linear" objective function does not output
    'predicted_label' or 'probabilities' (classification content). Invalid keys will be included
    in the response with an np.nan value.

    :param raw_predictions: output of xgboost predict (list of numpy objects)
    :param selected_keys: strings denoting selected keys (list of str)
    :param objective: objective xgboost training function (str)
    :param num_class: number of classes for multiclass classification (str, optional)
    :return: selected prediction (list of dict)
    """
    if objective not in VALID_OBJECTIVES:
        raise ValueError("Objective `{}` unsupported for selectable inference predictions."
                         .format(objective))

    valid_selected_keys = set(selected_keys).intersection(VALID_OBJECTIVES[objective])
    invalid_selected_keys = set(selected_keys).difference(VALID_OBJECTIVES[objective])
    if invalid_selected_keys:
        logging.warning("Selected key(s) {} incompatible for objective '{}'. "
                        "Please use list of compatible selectable inference predictions: {}"
                        .format(invalid_selected_keys, objective, VALID_OBJECTIVES[objective]))

    predictions = []
    for raw_prediction in raw_predictions:
        output = {}
        if PREDICTED_LABEL in valid_selected_keys:
            output[PREDICTED_LABEL] = _get_predicted_label(objective, raw_prediction)
        if LABELS in valid_selected_keys:
            output[LABELS] = _get_labels(objective, num_class=num_class)
        if PROBABILITY in valid_selected_keys:
            output[PROBABILITY] = _get_probability(objective, raw_prediction)
        if PROBABILITIES in valid_selected_keys:
            output[PROBABILITIES] = _get_probabilities(objective, raw_prediction)
        if RAW_SCORE in valid_selected_keys:
            output[RAW_SCORE] = _get_raw_score(objective, raw_prediction)
        if RAW_SCORES in valid_selected_keys:
            output[RAW_SCORES] = _get_raw_scores(objective, raw_prediction)
        if PREDICTED_SCORE in valid_selected_keys:
            output[PREDICTED_SCORE] = raw_prediction.item()
        if invalid_selected_keys:
            for invalid_selected_key in invalid_selected_keys:
                output[invalid_selected_key] = np.nan
        predictions.append(output)
    return predictions


def _encode_selected_predictions_csv(predictions, ordered_keys_list):
    """Encode predictions in csv format.

    For each prediction, the order of the content is determined by 'ordered_keys_list'.

    :param predictions: output of serve_utils.get_selected_predictions(...) (list of dict)
    :param ordered_keys_list: list of selected content keys (list of str)
    :return: predictions in csv response format (str)
    """
    def _generate_single_csv_line_selected_prediction(predictions, ordered_keys_list):
        """Generate a single csv line response for selectable inference predictions

        :param predictions: output of serve_utils.get_selected_predictions(...) (list of dict)
        :param ordered_keys_list: list of selected content keys (list of str)
        :return: yields a single csv row for each prediction (generator)
        """
        for single_prediction in predictions:
            values = []
            for key in ordered_keys_list:
                if isinstance(single_prediction[key], list):
                    value = '"{}"'.format(single_prediction[key])
                else:
                    value = str(single_prediction[key])
                values.append(value)
            yield ','.join(values)

    return '\n'.join(_generate_single_csv_line_selected_prediction(predictions, ordered_keys_list))


def _write_record(record, key, value):
    record.label[key].float32_tensor.values.extend(value)


def _encode_selected_predictions_recordio_protobuf(predictions):
    """Encode predictions in recordio-protobuf format.

    For each prediction, a new record is created. The content is populated under the "label" field
    of a record where the keys are derived from the selected content keys. Every value is encoded
    to a float32 tensor.

    :param predictions: output of serve_utils.get_selected_predictions(...) (list of dict)
    :return: predictions in recordio-protobuf response format (bytes)
    """
    record_bio = io.BytesIO()
    recordio_bio = io.BytesIO()
    record = Record()
    for item in predictions:
        for key in item.keys():
            value = item[key] if type(item[key]) is list else [item[key]]
            _write_record(record, key, value)
        record_bio.write(record.SerializeToString())
        record.Clear()
        _write_recordio(recordio_bio, record_bio.getvalue())
    return recordio_bio.getvalue()


def encode_selected_predictions(predictions, selected_content_keys, accept):
    """Encode the selected predictions and keys based on the given accept type.

    :param predictions: list of selected predictions (list of dict).
                        Output of serve_utils.get_selected_predictions(...)
                        See example below.
                        [{"predicted_label": 1, "probabilities": [0.4, 0.6]},
                         {"predicted_label": 0, "probabilities": [0.9, 0.1]}]
    :param selected_content_keys: list of selected content keys (list of str)
    :param accept: accept mime-type (str)
    :return: encoded content in accept
    """
    if accept == "application/json":
        return json.dumps({"predictions": predictions})
    if accept == "application/jsonlines":
        return json_to_jsonlines({"predictions": predictions})
    if accept == "application/x-recordio-protobuf":
        return _encode_selected_predictions_recordio_protobuf(predictions)
    if accept == "text/csv":
        csv_response = _encode_selected_predictions_csv(predictions, selected_content_keys)
        if SAGEMAKER_BATCH:
            return csv_response + '\n'
        return csv_response
    raise RuntimeError("Cannot encode selected predictions into accept type '{}'.".format(accept))
