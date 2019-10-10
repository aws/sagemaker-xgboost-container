import logging
from sagemaker_algorithm_toolkit.exceptions import UserError
from sagemaker_xgboost_container.checkpointing import load_checkpoint, save_checkpoint, SaveCheckpoint


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def add_checkpointing(callbacks, checkpoint_dir, num_boost_round, load_previous_checkpoint=True):
    """Convenience function for script mode.

    Users can add checkpointing in script mode by using this frunction to append to a list of callbacks.

    Example
    -------
    checkpoint_dir, num_boost_round = "/opt/ml/checkpoints", 10
    callbacks = []
    xgb_model, num_boost_round = add_checkpointing(callbacks, checkpoint_dir, num_boost_round)

    xgb.train(
        params,
        dtrain,
        callbacks=callbacks,
        xgb_model=xgb_model,
        num_boost_round=num_boost_round)
    """
    if any(isinstance(callback, SaveCheckpoint) for callback in callbacks):
        raise UserError("An instance of 'SaveCheckpoint' already exists in callbacks.")

    xgb_model, start_iteration = None, 0

    if load_previous_checkpoint is not True:
        callbacks.append(save_checkpoint(checkpoint_dir))
        return xgb_model, num_boost_round

    xgb_model, start_iteration = load_checkpoint(checkpoint_dir)
    if xgb_model is not None:
        logging.info("Checkpoint loaded from %s", xgb_model)
        logging.info("Resuming from iteration %s", start_iteration)

    callbacks.append(save_checkpoint(checkpoint_dir, start_iteration=start_iteration))

    num_boost_round -= start_iteration

    return xgb_model, num_boost_round
