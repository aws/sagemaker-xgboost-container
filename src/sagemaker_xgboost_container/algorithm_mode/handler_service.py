# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import
import numpy as np
import os
import pickle as pkl
from scipy.sparse import csr_matrix
import xgboost as xgb

from sagemaker_inference import content_types, default_inference_handler, encoder
from sagemaker_inference.default_handler_service import DefaultHandlerService

from sagemaker_xgboost_container.algorithm_mode.mms_transformer import Transformer
from sagemaker_xgboost_container import encoder as xgb_encoder
from sagemaker_xgboost_container.algorithm_mode.inference_errors import NoContentInferenceError, \
    UnsupportedMediaTypeInferenceError, ModelLoadInferenceError, BadRequestInferenceError


SAGEMAKER_BATCH = os.getenv("SAGEMAKER_BATCH")


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


class HandlerService(DefaultHandlerService):
    """Handler service that is executed by the model server.
    Determines specific default inference handlers to use based on the type MXNet model being used.
    This class extends ``DefaultHandlerService``, which define the following:
        - The ``handle`` method is invoked for all incoming inference requests to the model server.
        - The ``initialize`` method is invoked at model server start up.
    Based on: https://github.com/awslabs/mxnet-model-server/blob/master/docs/custom_service.md
    """

    class DefaultXGBoostAlgoModeInferenceHandler(default_inference_handler.DefaultInferenceHandler):

        def default_model_fn(self, model_dir):
            """Load a model. For XGBoost Framework, a default function to load a model is not provided.
            Users should provide customized model_fn() in script.
            Args:
                model_dir: a directory where model is saved.
            Returns:
                A XGBoost model.
                XGBoost model format type.
            """
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
            """Take request data and de-serializes the data into an object for prediction.
                When an InvokeEndpoint operation is made against an Endpoint running SageMaker model server,
                the model server receives two pieces of information:
                    - The request Content-Type, for example "application/json"
                    - The request data, which is at most 5 MB (5 * 1024 * 1024 bytes) in size.
                The input_fn is responsible to take the request data and pre-process it before prediction.
            Args:
                input_data (obj): the request data.
                content_type (str): the request Content-Type. XGBoost accepts CSV and LIBSVM.
            Returns:
                (obj): data ready for prediction. For XGBoost, this defaults to DMatrix.
            """
            if len(input_data) == 0:
                raise NoContentInferenceError()

            if content_type == "text/csv":
                try:
                    input_data = input_data.decode('utf-8')
                    payload = input_data.strip()
                    dtest = xgb_encoder.csv_to_dmatrix(payload, dtype=np.float)
                except Exception as e:
                    raise UnsupportedMediaTypeInferenceError("Loading csv data failed with "
                                                             "Exception, please ensure data "
                                                             "is in csv format: {} {}".format(type(e),
                                                                                              e))
            elif content_type == "text/x-libsvm" or content_type == 'text/libsvm':
                try:
                    # if not isinstance(input_data, str):
                    input_data = input_data.decode('utf-8')
                    payload = input_data.strip()
                    dtest = xgb.DMatrix(_get_sparse_matrix_from_libsvm(payload))
                except Exception as e:
                    raise UnsupportedMediaTypeInferenceError("Loading libsvm data failed with "
                                                             "Exception, please ensure data "
                                                             "is in libsvm format: {} {}".format(type(e),
                                                                                                 e))
            else:
                raise UnsupportedMediaTypeInferenceError("Content type must be either libsvm or csv.")

            return dtest, content_type

        def default_predict_fn(self, data, model):
            """A default predict_fn for XGBooost Framework. Calls a model on data deserialized in input_fn.
            Args:
                data: input data (DMatrix) for prediction deserialized by input_fn and data content type
                model: XGBoost model loaded in memory by model_fn, and xgboost model format
            Returns: a prediction
            """
            booster, model_format = model
            dtest, content_type = data
            try:
                return predict(booster, model_format, dtest, content_type)
            except Exception as e:
                raise BadRequestInferenceError(str(e))

        def default_output_fn(self, prediction, accept):
            """Return encoded prediction for the response.
            Args:
                prediction (obj): prediction returned by predict_fn .
                accept (str): accept content-type expected by the client.
            Returns:
                encoded response for MMS to return to client
            """
            try:
                if accept == content_types.CSV or accept == 'csv':
                    if SAGEMAKER_BATCH:
                        return_data = "\n".join(map(str, prediction.tolist())) + '\n'
                    else:
                        return_data = ",".join(map(str, prediction.tolist()))
                    encoded_prediction = return_data.encode("utf-8")
                elif accept == content_types.JSON or accept == 'json':
                    encoded_prediction = encoder.encode(prediction, accept)
                else:
                    raise ValueError("{} is not an accepted Accept type. Please choose one of the following:"
                                     " ['text/csv', 'application/json'].")
            except Exception as e:
                raise UnsupportedMediaTypeInferenceError(
                    "Encoding to accept type {} failed with exception: {}".format(accept,
                                                                                e))
            return encoded_prediction

    def __init__(self):
        transformer = Transformer(default_inference_handler=self.DefaultXGBoostAlgoModeInferenceHandler())
        super(HandlerService, self).__init__(transformer=transformer)
