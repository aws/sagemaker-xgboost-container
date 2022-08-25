import logging
from smdebug.xgboost import Hook
import xgboost as xgb


logger = logging.getLogger(__name__)

class SmeDebugHook(xgb.callback.TrainingCallback, Hook):
    """wrap sagemaker debug hook to TrainingCallback style
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
        