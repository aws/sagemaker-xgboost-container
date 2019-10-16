import json
import logging
from tornasole.xgboost import TornasoleHook


logger = logging.getLogger(__name__)


def add_tornasole_hook(callbacks, hyperparameters, train_dmatrix,
                       val_dmatrix=None, json_config_path=None):
    """Add a tornasole hook to a list of callbacks.

    :param callbacks: List of callback functions.
    :param hyperparameters: Dict of hyperparamters.
                            Same as `params` in xgb.train(params, dtrain).
    :param val_dmatrix: Validation data set.
    :param json_config_path: If specified, this json config will be used
                             instead of default config file.
    """
    try:
        tornasole_hook = TornasoleHook.hook_from_config(json_config_path)
        tornasole_hook.hyperparameters = hyperparameters
        tornasole_hook.train_data = train_dmatrix
        if val_dmatrix is not None:
            tornasole_hook.validation_data = val_dmatrix
        logging.info("Tornasole hook created from config")
    except Exception as e:
        logging.debug("Failed to create Toransole hook", e)
        return
    callbacks.append(tornasole_hook)
