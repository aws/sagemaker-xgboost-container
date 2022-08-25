import logging
from smdebug.xgboost import Hook
import xgboost as xgb


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
    def __init__(self, hyperparameters, train_dmatrix,
                 val_dmatrix=None):
        try:
            Hook.__init__(self)
            self.hyperparameters = hyperparameters
            self.train_data = train_dmatrix
            if val_dmatrix is not None:
                self.validation_data = val_dmatrix
            logging.info("Debug hook initialized")
        except Exception as e:
            logging.debug("Failed to create debug hook", e)
            return
