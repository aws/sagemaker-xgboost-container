import logging

import xgboost as xgb
from smdebug.xgboost import Hook

logger = logging.getLogger(__name__)


class SmeDebugHook(xgb.callback.TrainingCallback, Hook):
    """Mix-in callback class. smedebug.xgboost.Hook uses legacy callback style
    and since XGB-3.0.0 mixing legacy callback instances with new TrainingCallback
    instances is not allowed.

    See: https://github.com/dmlc/xgboost/blob/v1.3.0/python-package/xgboost/training.py#L92-L93

    :param hyperparameters: Dict of hyperparamters.
                            Same as `params` in xgb.train(params, dtrain).
    :param train_dmatrix: Training data set.
    :param val_dmatrix: Validation data set.
    """

    def __init__(self, json_config_path, hyperparameters, train_dmatrix, val_dmatrix):
        self = self.hook_from_config(json_config_path)
        self.hyperparameters = hyperparameters
        self.train_data = train_dmatrix
        if val_dmatrix is not None:
            self.validation_data = val_dmatrix


def add_debugging(callbacks, hyperparameters, train_dmatrix, val_dmatrix=None, json_config_path=None):
    """Add a sagemaker debug hook to a list of callbacks.

    :param callbacks: List of callback functions.
    :param hyperparameters: Dict of hyperparamters.
                            Same as `params` in xgb.train(params, dtrain).
    :param train_dmatrix: Training data set.
    :param val_dmatrix: Validation data set.
    :param json_config_path: If specified, this json config will be used
                             instead of default config file.
    """
    try:
        hook = SmeDebugHook(json_config_path, hyperparameters, train_dmatrix, val_dmatrix)
        logging.info("Debug hook created from config")
    except Exception as e:
        logging.debug("Failed to create debug hook", e)
    else:
        callbacks.append(hook)
