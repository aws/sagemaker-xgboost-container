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
import cgi
import http.client
import json
import multiprocessing
import os
import signal
import sys

from gunicorn.six import iteritems
import flask
import gunicorn.app.base
from sagemaker_containers import _content_types

from sagemaker_containers.beta.framework import encoders
from sagemaker_xgboost_container.algorithm_mode import integration
from sagemaker_xgboost_container.algorithm_mode import serve_utils
from sagemaker_xgboost_container.constants import sm_env_constants


SAGEMAKER_BATCH = os.getenv(sm_env_constants.SAGEMAKER_BATCH)
SUPPORTED_ACCEPTS = ["application/json", "application/jsonlines", "application/x-recordio-protobuf", "text/csv"]
logging = integration.setup_main_logger(__name__)


PARSED_MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", '6291456'))


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
    MODEL_PATH = os.getenv(sm_env_constants.SM_MODEL_DIR)
    MAX_CONTENT_LENGTH = PARSED_MAX_CONTENT_LENGTH
    app = flask.Flask(__name__)
    booster = None
    format = None
    config_json = None
    objective = None

    @classmethod
    def load_model(cls, ensemble=True):
        if cls.booster is None:
            cls.booster, cls.format = serve_utils.get_loaded_booster(ScoringService.MODEL_PATH, ensemble)
            cls.get_config_json()
        return cls.format

    @classmethod
    def predict(cls, data, content_type='text/x-libsvm', model_format='pkl_format'):
        return serve_utils.predict(cls.booster, model_format, data, content_type, cls.objective)

    @classmethod
    def get_config_json(cls):
        """Gets the internal parameter configuration of a fitted XGBoost booster.

        :return: xgboost booster's internal configuration (dict)
        """
        if cls.config_json is None:
            booster = cls.booster[0] if isinstance(cls.booster, list) else cls.booster
            cls.config_json = json.loads(booster.save_config())
            cls.objective = cls.config_json["learner"]["objective"]["name"]
            logging.info("Model objective : {}".format(cls.objective))
        return cls.config_json

    @staticmethod
    def post_worker_init(worker):
        """
        Gunicorn server hook http://docs.gunicorn.org/en/stable/settings.html#post-worker-init
        :param worker: Gunicorn worker

        # Model is being loaded per worker because xgboost predict is not thread safe.
        # See https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/core.py#L997
        """
        try:
            load_model()
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


def load_model():
    return ScoringService.load_model(ensemble=serve_utils.is_ensemble_enabled())


@ScoringService.app.route("/ping", methods=["GET"])
def ping():
    # TODO: implement health checks
    load_model()
    return flask.Response(status=http.client.OK)


@ScoringService.app.route("/execution-parameters", methods=["GET"])
def execution_parameters():
    try:
        # TODO: implement logics to find optimal/sub-optimal parameters
        parameters = {
            "MaxConcurrentTransforms": number_of_workers(),
            "BatchStrategy": "MULTI_RECORD",
            "MaxPayloadInMB": int(PARSED_MAX_CONTENT_LENGTH / (1024 ** 2))
        }
    except Exception as e:
        return flask.Response(response="Unable to determine execution parameters: %s" % e,
                              status=http.client.INTERNAL_SERVER_ERROR)

    response_text = json.dumps(parameters)
    return flask.Response(response=response_text, status=http.client.OK, mimetype="application/json")


def _parse_accept(request):
    """Get the accept type for a given request.

    Valid accept types are "application/json", "application/jsonlines", "application/x-recordio-protobuf",
    and "text/csv". If no accept type is set, use the value in SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT.

    :param request: flask request
    :return: parsed accept type
    """
    accept, _ = cgi.parse_header(request.headers.get("accept", ""))
    if not accept or accept == "*/*":
        return os.getenv(sm_env_constants.SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT, "text/csv")
    if accept.lower() not in SUPPORTED_ACCEPTS:
        raise ValueError("Accept type {} is not supported. Please use supported accept types: {}."
                         .format(accept, SUPPORTED_ACCEPTS))
    return accept.lower()


def _handle_selectable_inference_response(predictions, accept):
    """Retrieve the additional prediction data for selectable inference mode.

    :param predictions: output of xgboost predict (list of numpy objects)
    :param accept: requested accept type (str)
    :return: flask response with encoded predictions
    """
    try:
        config = ScoringService.get_config_json()
        objective = config['learner']['objective']['name']
        num_class = config['learner']['learner_model_param'].get('num_class', '')
        selected_content_keys = serve_utils.get_selected_output_keys()

        selected_content = serve_utils.get_selected_predictions(predictions, selected_content_keys, objective,
                                                                num_class=num_class)

        response = serve_utils.encode_selected_predictions(selected_content, selected_content_keys, accept)
    except Exception as e:
        logging.exception(e)
        return flask.Response(response=str(e), status=http.client.INTERNAL_SERVER_ERROR)

    return flask.Response(response=response, status=http.client.OK, mimetype=accept)


@ScoringService.app.route("/invocations", methods=["POST"])
def invocations():
    payload = flask.request.data
    if len(payload) == 0:
        return flask.Response(response="", status=http.client.NO_CONTENT)

    try:
        dtest, content_type = serve_utils.parse_content_data(payload, flask.request.content_type)
    except Exception as e:
        logging.exception(e)
        return flask.Response(response=str(e), status=http.client.UNSUPPORTED_MEDIA_TYPE)

    try:
        format = load_model()
    except Exception as e:
        logging.exception(e)
        return flask.Response(response="Unable to load model: %s" % e, status=http.client.INTERNAL_SERVER_ERROR)

    try:
        preds = ScoringService.predict(data=dtest, content_type=content_type, model_format=format)
    except Exception as e:
        logging.exception(e)
        return flask.Response(response="Unable to evaluate payload provided: %s" % e, status=http.client.BAD_REQUEST)

    try:
        accept = _parse_accept(flask.request)
    except Exception as e:
        logging.exception(e)
        return flask.Response(response=str(e), status=http.client.NOT_ACCEPTABLE)

    if serve_utils.is_selectable_inference_output():
        return _handle_selectable_inference_response(preds, accept)

    preds_list = preds.tolist()
    if SAGEMAKER_BATCH:
        return_data = "\n".join(map(str, preds_list)) + '\n'
    else:
        if accept == _content_types.JSON:
            return_data = serve_utils.encode_predictions_as_json(preds_list)
        else:
            return_data = encoders.encode(preds_list, accept)

    return flask.Response(response=return_data, status=http.client.OK, mimetype=accept)


if __name__ == '__main__':
    ScoringService.start()
