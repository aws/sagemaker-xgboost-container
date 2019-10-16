import csv
import http
import logging
import numpy as np
import os
import pickle as pkl
import time
from scipy.sparse import csr_matrix
import xgboost as xgb

from sagemaker_containers.beta.framework import (
    encoders, env, modules, transformer, worker)
from sagemaker_inference import content_types, decoder, default_inference_handler, errors

from sagemaker_algorithm_toolkit.exceptions import BaseInferenceToolkitError
from sagemaker_algorithm_toolkit.exceptions import NoContentInferenceError, UnsupportedMediaTypeInferenceError, \
    ModelLoadInferenceError, BadRequestInferenceError

from sagemaker_xgboost_container import encoder
from sagemaker_xgboost_container.algorithm_mode import serve_utils


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


def parse_content_data(input_data, content_type):
    if content_type == "text/csv":
        try:
            payload = input_data.strip()
            dtest = encoder.csv_to_dmatrix(payload, dtype=np.float)
        except Exception as e:
            raise RuntimeError("Loading csv data failed with Exception, "
                               "please ensure data is in csv format:\n {}\n {}".format(type(e), e))
    elif content_type == "text/x-libsvm" or content_type == 'text/libsvm':
        try:
            payload = input_data.strip()
            dtest = xgb.DMatrix(_get_sparse_matrix_from_libsvm(payload))
        except Exception as e:
            raise RuntimeError("Loading libsvm data failed with Exception, "
                               "please ensure data is in libsvm format:\n {}\n {}".format(type(e), e))
    else:
        raise RuntimeError("Content type must be either libsvm or csv.")

    return dtest


def predict(booster, model_format, dtest, content_type):
    if model_format == 'pkl_format':
        x = len(booster.feature_names)
        y = len(dtest.feature_names)

        if content_type == 'text/x-libsvm' or content_type == 'text/libsvm':
            if y > x + 1:
                raise ValueError('Feature size of libsvm inference data {} is larger than '
                                 'feature size of trained model {}.'.format(y, x))
        elif content_type == 'text/csv':
            if not ((x == y) or (x == y + 1)):
                raise ValueError('Feature size of csv inference data {} is not consistent '
                                 'with feature size of trained model {}'.format(y, x))
        else:
            raise ValueError('Content type {} is not supported'.format(content_type))
    return booster.predict(dtest,
                           ntree_limit=getattr(booster, "best_ntree_limit", 0),
                           validate_features=False)


class DefaultXGBoostAlgoModeInferenceHandler(default_inference_handler.DefaultInferenceHandler):
    VALID_CONTENT_TYPES = (content_types.JSON, content_types.NPY)

    def default_model_fn(self, model_dir):
        model_file = os.listdir(model_dir)[0]
        try:
            booster = pkl.load(open(os.path.join(model_dir, model_file), 'rb'))
            format = 'pkl_format'
        except Exception as exp_pkl:
            try:
                booster = xgb.Booster()
                booster.load_model(os.path.join(model_dir, model_file))
                format = 'xgb_format'
            except Exception as exp_xgb:
                raise ModelLoadInferenceError("Unable to load model: %s %s", exp_pkl, exp_xgb)
        booster.set_param('nthread', 1)
        return booster, format

    def default_input_fn(self, input_data, content_type):
        """A default input_fn that can handle JSON, CSV and NPZ formats.
        Args:
            input_data: the request payload serialized in the content_type format
            content_type: the request content_type
        Returns: input_data deserialized into torch.FloatTensor or torch.cuda.FloatTensor depending if cuda is available.
        """
        if len(input_data) == 0:
            raise NoContentInferenceError()

        if content_type == "text/csv":
            try:
                if not isinstance(input_data, str):
                    input_data = input_data.decode('utf-8')
                payload = input_data.strip()
                dtest = encoder.csv_to_dmatrix(payload, dtype=np.float)
            except Exception as e:
                raise UnsupportedMediaTypeInferenceError("Loading csv data failed with Exception, "
                                                         "please ensure data is in csv format: {} {}".format(type(e),
                                                                                                             e))
        elif content_type == "text/x-libsvm" or content_type == 'text/libsvm':
            try:
                if not isinstance(input_data, str):
                    input_data = input_data.decode('utf-8')
                payload = input_data.strip()
                dtest = xgb.DMatrix(_get_sparse_matrix_from_libsvm(payload))
            except Exception as e:
                raise UnsupportedMediaTypeInferenceError("Loading libsvm data failed with Exception, "
                                                         "please ensure data is in libsvm format: {} {}".format(type(e),
                                                                                                                e))
        else:
            raise UnsupportedMediaTypeInferenceError("Content type must be either libsvm or csv.")

        return dtest, content_type

    def default_predict_fn(self, data, model):
        booster, model_format = model
        dtest, content_type = data
        try:
            return predict(booster, model_format, dtest, content_type)
        except Exception as e:
            raise BadRequestInferenceError(str(e))

    def default_output_fn(self, prediction, accept):
        """A default output_fn for PyTorch. Serializes predictions from predict_fn to JSON, CSV or NPZ format.
        Args:
            prediction: a prediction result from predict_fn
            accept: type which the output data needs to be serialized
        Returns: output data serialized
        """
        return encoders.encode(prediction, accept)
