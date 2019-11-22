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
import http.client
import json
import multiprocessing
import os
import pickle as pkl
import signal
import sys

from gunicorn.six import iteritems
from scipy.sparse import csr_matrix
import flask
import gunicorn.app.base
import numpy as np
import xgboost as xgb

from sagemaker_xgboost_container import encoder
from sagemaker_xgboost_container.constants import sm_env_constants
from sagemaker_xgboost_container.algorithm_mode import integration
from sagemaker_xgboost_container.algorithm_mode import serve_utils


SAGEMAKER_BATCH = os.getenv("SAGEMAKER_BATCH")
logging = integration.setup_main_logger(__name__)


def number_of_workers():
    return multiprocessing.cpu_count()


class GunicornApplication(gunicorn.app.base.BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super(GunicornApplication, self).__init__()

    def load_config(self):
        for key, value in iteritems(self.options):
            key = key.lower()
            if key in self.cfg.settings and value is not None:
                self.cfg.set(key, value)

    def load(self):
        return self.application


class ScoringService(object):
    PORT = os.getenv("SAGEMAKER_BIND_TO_PORT", 8080)
    # NOTE: 6 MB max content length
    MAX_CONTENT_LENGTH = os.getenv("MAX_CONTENT_LENGTH", 6 * 1024 * 1024)

    MODEL_PATH = os.getenv(sm_env_constants.SM_MODEL_DIR)
    app = flask.Flask(__name__)
    booster = None
    format = None

    @classmethod
    def load_model(cls):
        if cls.booster is None:
            try:
                model_file = os.listdir(ScoringService.MODEL_PATH)[0]
                cls.booster = pkl.load(open(os.path.join(ScoringService.MODEL_PATH, model_file), 'rb'))
                cls.format = 'pkl_format'
            except Exception as exp_pkl:
                try:
                    model_file = os.listdir(ScoringService.MODEL_PATH)[0]
                    cls.booster = xgb.Booster()
                    cls.booster.load_model(os.path.join(ScoringService.MODEL_PATH, model_file))
                    cls.format = 'xgb_format'
                except Exception as exp_xgb:
                    raise RuntimeError("Unable to load model: %s %s", exp_pkl, exp_xgb)
        cls.booster.set_param('nthread', 1)
        return cls.format

    @classmethod
    def predict(cls, data, content_type='text/x-libsvm', model_format='pkl_format'):
        if model_format == 'pkl_format':
            x = len(cls.booster.feature_names)
            y = len(data.feature_names)

            if content_type == 'text/x-libsvm' or content_type == 'text/libsvm':
                if y > x + 1:
                    raise ValueError('Feature size of libsvm inference data {} is larger than '
                                     'feature size of trained model {}.'.format(y, x))
            elif content_type == 'text/csv':
                if not ((x == y) or (x == y + 1)):
                    raise ValueError('Feature size of csv inference data {} is not consistent '
                                     'with feature size of trained model {}'.format(y, x))
            elif content_type == 'application/x-recordio-protobuf':
                if not ((x == y) or (x == y + 1)):
                    raise ValueError('Feature size of recordio-protobuf inference data {} is not consistent '
                                     'with feature size of trained model {}.'.format(y, x))
            else:
                raise ValueError('Content type {} is not supported'.format(content_type))
        return cls.booster.predict(data,
                                   ntree_limit=getattr(cls.booster, "best_ntree_limit", 0),
                                   validate_features=False)

    @staticmethod
    def post_worker_init(worker):
        """
        Gunicorn server hook http://docs.gunicorn.org/en/stable/settings.html#post-worker-init
        :param worker: Gunicorn worker

        # Model is being loaded per worker because xgboost predict is not thread safe.
        # See https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/core.py#L997
        """

        try:
            ScoringService.load_model()
        except Exception as e:
            logging.exception(e)
            sys.exit(1)

        logging.info("Model loaded successfully for worker : {}".format(worker.pid))

    @staticmethod
    def start():
        # NOTE: Stop Flask application when SIGTERM is received as a result of "docker stop" command.
        signal.signal(signal.SIGTERM, ScoringService.stop)

        ScoringService.app.config["MAX_CONTENT_LENGTH"] = ScoringService.MAX_CONTENT_LENGTH
        options = {
            "bind": "{}:{}".format("0.0.0.0", ScoringService.PORT),
            "workers": number_of_workers(),
            "worker_class": "gevent",
            "keepalive": 60,
            "post_worker_init": ScoringService.post_worker_init,
        }
        GunicornApplication(ScoringService.app, options).run()

    @staticmethod
    def stop():
        ScoringService.app.shutdown()

    @staticmethod
    def csdk_start():
        ScoringService.app.config["MAX_CONTENT_LENGTH"] = ScoringService.MAX_CONTENT_LENGTH
        return ScoringService.app


@ScoringService.app.route("/ping", methods=["GET"])
def ping():
    # TODO: implement health checks
    return flask.Response(status=http.client.OK)


@ScoringService.app.route("/execution-parameters", methods=["GET"])
def execution_parameters():
    try:
        # TODO: implement logics to find optimal/sub-optimal parameters
        parameters = {
            "MaxConcurrentTransforms": number_of_workers(),
            "BatchStrategy": "MULTI_RECORD",
            "MaxPayloadInMB": 6
        }
    except Exception as e:
        return flask.Response(response="Unable to determine execution parameters: %s" % e,
                              status=http.client.INTERNAL_SERVER_ERROR)

    response_text = json.dumps(parameters)
    return flask.Response(response=response_text, status=http.client.OK, mimetype="application/json")


# FIXME: https://github.com/aws/sagemaker-xgboost-container/issues/12
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


def _parse_content_data(request):
    dtest = None
    content_type = serve_utils.get_content_type(request)
    payload = request.data
    if content_type == "text/csv":
        try:
            decoded_payload = payload.strip().decode("utf-8")
            dtest = encoder.csv_to_dmatrix(decoded_payload, dtype=np.float)
        except Exception as e:
            raise RuntimeError("Loading csv data failed with Exception, "
                               "please ensure data is in csv format:\n {}\n {}".format(type(e), e))
    elif content_type == "text/x-libsvm" or content_type == 'text/libsvm':
        try:
            decoded_payload = payload.strip().decode("utf-8")
            dtest = xgb.DMatrix(_get_sparse_matrix_from_libsvm(decoded_payload))
        except Exception as e:
            raise RuntimeError("Loading libsvm data failed with Exception, "
                               "please ensure data is in libsvm format:\n {}\n {}".format(type(e), e))
    elif content_type == "application/x-recordio-protobuf":
        try:
            dtest = encoder.recordio_protobuf_to_dmatrix(payload)
        except Exception as e:
            raise RuntimeError("Loading recordio-protobuf data failed with "
                               "Exception, please ensure data is in "
                               "recordio-protobuf format: {} {}".format(type(e), e))

    return dtest, content_type


@ScoringService.app.route("/invocations", methods=["POST"])
def invocations():
    payload = flask.request.data
    if len(payload) == 0:
        return flask.Response(response="", status=http.client.NO_CONTENT)

    try:
        dtest, content_type = _parse_content_data(flask.request)
    except Exception as e:
        logging.exception(e)
        return flask.Response(response=str(e), status=http.client.UNSUPPORTED_MEDIA_TYPE)

    try:
        format = ScoringService.load_model()
    except Exception as e:
        logging.exception(e)
        return flask.Response(response="Unable to load model: %s" % e, status=http.client.INTERNAL_SERVER_ERROR)

    try:
        preds = ScoringService.predict(data=dtest, content_type=content_type, model_format=format)
    except Exception as e:
        logging.exception(e)
        return flask.Response(response="Unable to evaluate payload provided: %s" % e, status=http.client.BAD_REQUEST)

    return_data = ",".join(map(str, preds.tolist()))
    if SAGEMAKER_BATCH:
        return_data = "\n".join(map(str, preds.tolist())) + '\n'

    return flask.Response(response=return_data, status=http.client.OK, mimetype="text/csv")


if __name__ == '__main__':
    ScoringService.start()
