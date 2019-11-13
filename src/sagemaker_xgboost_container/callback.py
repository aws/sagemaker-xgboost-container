import logging
from smdebug.xgboost import Hook


logger = logging.getLogger(__name__)


def add_debugging(callbacks, hyperparameters, train_dmatrix,
                  val_dmatrix=None, json_config_path=None):
    """Add a sagemaker debug hook to a list of callbacks.

    :param callbacks: List of callback functions.
    :param hyperparameters: Dict of hyperparamters.
                            Same as `params` in xgb.train(params, dtrain).
    :param val_dmatrix: Validation data set.
    :param json_config_path: If specified, this json config will be used
                             instead of default config file.
    """
    try:
        hook = Hook.hook_from_config(json_config_path)
        hook.hyperparameters = hyperparameters
        hook.train_data = train_dmatrix
        if val_dmatrix is not None:
            hook.validation_data = val_dmatrix
        logging.info("Debug hook created from config")
    except Exception as e:
        logging.debug("Failed to create debug hook", e)
        return
    callbacks.append(hook)
