import logging
import os
import signal
import xgboost as xgb

from sagemaker_xgboost_container import checkpointing
from sagemaker_xgboost_container.algorithm_mode import train_utils
from sagemaker_xgboost_container.constants.xgb_constants import MODEL_NAME, XGB_MAXIMIZE_METRICS
from smdebug.xgboost import Hook

logger = logging.getLogger(__name__)


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
        hook = Hook.hook_from_config(json_config_path)
        hook.hyperparameters = hyperparameters
        hook.train_data = train_dmatrix
        if val_dmatrix is not None:
            hook.validation_data = val_dmatrix
        callbacks.append(hook)
        logging.info("Debug hook created from config")
    except Exception as e:
        logging.debug("Failed to create debug hook", e)
        return


def add_sigterm_handler(model_dir, is_master):
    """Stop training and cleanup model directory when SIGTERM is received.

    Model directory is only cleaned if is_master is True. Otherwise program terminates.

    :param model_dir: Directory where model is saved
    :param is_master: True if single node training, or the current node is the master node in distributed training
    """

    def _terminate():
        os._exit(0)

    def _cleanup_files(signo, frame):
        if is_master:
            train_utils.cleanup_dir(model_dir, MODEL_NAME)

        _terminate()

    signal.signal(signal.SIGTERM, _cleanup_files)


def get_callbacks(
    model_dir,
    checkpoint_dir,
    early_stopping_data_name,
    early_stopping_metric,
    early_stopping_rounds,
    save_model_on_termination,
    is_master,
    fold=None,
):
    if checkpoint_dir and fold is not None:
        checkpoint_dir = os.path.join(checkpoint_dir, f"model-{fold}")

    # Set callbacks
    xgb_model, iteration = checkpointing.load_checkpoint(checkpoint_dir)
    if xgb_model is not None:
        if fold is not None:
            xgb_model = f"{xgb_model}-{fold}"
        logging.info("Checkpoint loaded from %s", xgb_model)
        logging.info("Resuming from iteration %s", iteration)

    callbacks = []
    callbacks.append(xgb.callback.EvaluationMonitor())

    if checkpoint_dir and is_master:
        save_checkpoint = xgb.callback.TrainingCheckPoint(
            directory=checkpoint_dir, interval=iteration, name=checkpointing.CHECKPOINT_FILENAME
        )
        callbacks.append(save_checkpoint)

    logging.info(f"CALLBACK_SETUP_DEBUG: save_model_on_termination={save_model_on_termination}, is_master={is_master}")

    if save_model_on_termination == "true" and is_master:
        logging.info("CALLBACK_ADDING: Adding SaveIntermediateModelCallBack on master")
        model_name = f"{MODEL_NAME}-{fold}" if fold is not None else MODEL_NAME
        save_intermediate_model = checkpointing.SaveIntermediateModelCallBack(model_dir, model_name, is_master)
        callbacks.append(save_intermediate_model)
        add_sigterm_handler(model_dir, is_master)
    else:
        logging.info(f"CALLBACK_SKIPPING save_model_on_termination={save_model_on_termination}, is_master={is_master})")

    if early_stopping_data_name and early_stopping_metric and early_stopping_rounds:
        maximize = early_stopping_metric in XGB_MAXIMIZE_METRICS
        early_stop = xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds,
            data_name=early_stopping_data_name,
            metric_name=early_stopping_metric,
            maximize=maximize,
            save_best=is_master,
        )
        callbacks.append(early_stop)

    return xgb_model, iteration, callbacks
