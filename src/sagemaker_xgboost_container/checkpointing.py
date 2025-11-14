import logging
import os
import queue
import re
import tempfile
import threading

import xgboost as xgb
from typing import Optional

# from xgboost import rabit
from xgboost.callback import EvaluationMonitor
from xgboost.core import XGBoostError

TEMP_FILE_SUFFIX = ".sagemaker-ignore"
FILE_LOCK_SUFFIX = ".sagemaker-uploading"
FILE_SAFE_SUFFIX = ".sagemaker-uploaded"

CHECKPOINT_FILENAME = "xgboost-checkpoint"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train(train_args, checkpoint_dir):
    """Convenience function for script mode.

    Instead of running xgb.train(params, dtrain, ...), users can enable
    checkpointing in script mode by creating a dictionary of xgb.train
    arguments:

    train_args = dict(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_round,
        evals=[(dtrain, 'train'), (dtest, 'test')]
    )

    and calling:

    bst = checkpointing.train(train_args)
    """
    train_args = train_args.copy()

    xgb_model, start_iteration = load_checkpoint(checkpoint_dir)

    # xgboost's default value for num_boost_round is 10.
    # https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.training
    # If num_boost_round <= 0, xgb.train() doesn't actually train and
    # immediately returns a Booster object.
    train_args["num_boost_round"] = train_args.get("num_boost_round", 10) - start_iteration

    if xgb_model is not None:
        logging.info("Checkpoint loaded from %s", xgb_model)
        logging.info("Resuming from iteration %s", start_iteration)

    callbacks = train_args.get("callbacks", [])
    callbacks.append(
        print_checkpointed_evaluation(start_iteration=start_iteration, end_iteration=train_args["num_boost_round"])
    )
    callbacks.append(
        save_checkpoint(
            checkpoint_dir,
            start_iteration=start_iteration,
            iteration=start_iteration,
            end_iteration=train_args["num_boost_round"],
        )
    )

    train_args["verbose_eval"] = False  # suppress xgboost's print_evaluation()
    train_args["xgb_model"] = xgb_model
    train_args["callbacks"] = callbacks

    booster = xgb.train(**train_args)

    return booster


class PrintCheckpoint(xgb.callback.TrainingCallback):
    """Create a callback that print evaluation result every period iteration.

    This function was modified from https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/callback.py
    The only difference between the following function and the original function in xgboost.callback
    is the additional 'start_iteration' parameter.

    We print the evaluation results every **period** iterations
    and on the first and the last iterations.

    Attributes
    ----------
    period : int
    The period to log the evaluation results
    show_stdv : bool, optional
    Whether show stdv if provided
    start_iteration: int, optioonal
    Used for offsetting the iteratoin number that appears at the beginning of each evaluation result in the logs.
    """

    def __init__(self, end_iteration, iteration=0, rank=0, period=1, show_stdv=True, start_iteration=0):
        self.period = period
        self.show_stdv = show_stdv
        self.start_iteration = start_iteration
        self.rank = rank
        self.iteration = iteration
        self.end_iteration = end_iteration

    def __call__(self, model, epoch=0, evals_log=None):
        return self.after_iteration(model, epoch, evals_log)

    def after_iteration(self, model, epoch=0, evals_log=None):
        if self.rank != 0 or (not evals_log) or self.period is False or self.period == 0:
            return
        i = self.iteration
        if i % self.period == 0 or i + 1 == self.start_iteration or i + 1 == self.end_iteration:
            evaluation_monitor = EvaluationMonitor(self.rank, self.period, self.show_stdv)
            msg: str = ""
            for data, metric in evals_log.items():
                for metric_name, log in metric.items():
                    stdv: Optional[float] = None
                    if isinstance(log[-1], tuple):
                        score = log[-1][0]
                        stdv = log[-1][1]
                    else:
                        score = log[-1]
                    msg += evaluation_monitor._fmt_metric(data, metric_name, score, stdv)
            msg += "\n"
            # rabit.tracker_print("[%d]\t%s\n" % (i + self.start_iteration, msg))


def print_checkpointed_evaluation(end_iteration, iteration=0, rank=0, period=1, show_stdv=True, start_iteration=0):
    """A callback function that print evaluation result every period iteration.

    This is a wrapper function around PrintCheckpoint.
    For details, see PrintCheckpoint.
    """
    return PrintCheckpoint(end_iteration, iteration, rank, period, show_stdv, start_iteration)


def load_checkpoint(checkpoint_dir, max_try=5):
    """
    :param checkpoint_dir: e.g., /opt/ml/checkpoints
    :param max_try: number of times to try loading checkpoint before giving up.
    :return xgb_model: file path of stored xgb model. None if no checkpoint.
    :return iteration: iterations completed before last checkpoint.
    """
    if not checkpoint_dir or not os.path.exists(checkpoint_dir):
        return None, 0

    regex = r"^{0}\.[0-9]+$".format(CHECKPOINT_FILENAME)
    checkpoints = [f for f in os.listdir(checkpoint_dir) if re.match(regex, f)]
    if not checkpoints:
        return None, 0
    _sort_checkpoints(checkpoints)

    xgb_model, iteration = None, 0

    for _ in range(max_try):
        try:
            latest_checkpoint = checkpoints.pop()
            xgb_model = os.path.join(checkpoint_dir, latest_checkpoint)
            filename, extension = latest_checkpoint.split(".")
            iteration = int(extension) + 1
            break
        except XGBoostError:
            logging.debug("Wrong checkpoint model format %s", latest_checkpoint)

    return xgb_model, iteration


def _sort_checkpoints(checkpoint_files):
    checkpoint_files.sort(key=lambda x: int(x.split(".")[1]))
    return checkpoint_files


def save_checkpoint(
    checkpoint_dir, start_iteration=0, max_to_keep=5, num_round=None, rank=0, iteration=0, end_iteration=None
):
    """A callback function that saves checkpoints to disk.

    This is a wrapper function around SaveCheckpoint.
    For details, see SaveCheckpoint.
    """
    return SaveCheckpointCallBack(
        checkpoint_dir=checkpoint_dir,
        start_iteration=start_iteration,
        max_to_keep=max_to_keep,
        num_round=num_round,
        iteration=iteration,
        end_iteration=end_iteration,
    )


class SaveCheckpointCallBack(xgb.callback.TrainingCallback):
    """Create a callback that saves checkpoints to disk.

    The main purpose of this class is to support checkpointing for managed spot
    training:
    https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html.
    Since spot instances can be interrupted at anytime, we need to be able to
    save checkpoints during training, and we also need to be able to resume
    from the last checkpoint when the training job is restarted.

    We save each checkpoint to different files at the end of each iteration by
    appending the iteration number to the file name, e.g., xgboost-checkpont.1,
    xgboost-checkpont.2, and so on. These files are written to `checkpoint_dir`,

    Since saving one checkpoint per iteration could result in a large number of
    files to save in S3 and download when spot instances are resumed, we retain
    only the `max_to_keep` (5 by default) most recent checkpoints in the
    directory. This is accomplished by a background thread that deletes all
    checkpoints older than 5 most recent checkpoints (the number of files to
    keep is somewhat arbitrary, choosing the optimal number of files to keep is
    left for future work). Note that when a file is being uploaded by SM, it
    will create a marker file (file name + .sagemaker-uploading) to indicate
    that the file is being uploaded. SM will also create another marker file
    (file name + .sagemaker-uploaded) when the upload is completed. Thus, the
    background will skip deleting a file and try again later if there is a
    marker file <filename>.sagemaker-uploading and only attempt to delete a
    file when the marker file <filename>.sagemaker-uploaded is present.

    Attributes:
        checkpoint_dir: indicates the path to the directory where checkpoints
            will be saved.  Defaults to /opt/ml/checkpoints on SageMaker.
        max_to_keep: indicates the maximum number of recent checkpoint files to
            keep.  As new files are created, older files are deleted.  Defaults
            to 5 (that is, the 5 most recent checkpoint files are kept.)
        start_iteration: indicates the round at which the current training
            started. If xgb_model was loaded from a previous checkpoint, this
            will be greater than 0 (that is, if the previous training ended
            after round 19, start_iteration will be 20).
        num_round: (optional) indicates the number of boosting rounds.

        Example:
            >>> save_checkpoint = SaveCheckpoint("/opt/ml/checkpoints")
            >>> xgboost.train(prams, dtrain, callbacks=[save_checkpoint])
    """

    SENTINEL = None

    def __init__(
        self, checkpoint_dir, start_iteration=0, max_to_keep=5, num_round=None, rank=0, iteration=0, end_iteration=None
    ):
        """Init SaveCheckpoint with checkpoint_dir"""
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        self.start_iteration = start_iteration
        self.num_round = num_round
        self.rank = rank
        self.iteration = iteration
        self.end_iteration = end_iteration

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.previous_checkpoints = [os.path.join(self.checkpoint_dir, f) for f in os.listdir(self.checkpoint_dir)]

        self.thread = None
        self.delete_queue = queue.Queue()

        self.start()

    def __call__(self, model, epoch=0, evals_log=None):
        """Make the class callable since it is meant be used as a callback"""
        return self.after_iteration(model, epoch, evals_log)

    def format_path(self, iteration):
        """Return a file path to checkpoint given a iteration number"""
        filename = "{}.{}".format(CHECKPOINT_FILENAME, iteration)
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        return checkpoint_path

    def after_iteration(self, model, epoch=0, evals_log=None) -> bool:
        # rank: master node has rank 0.
        # iteration: current boosting round
        # end_iteration: round # when training will end. this is always num_round + 1.
        # model: model object
        if self.rank != 0:
            logger.debug("Not master (rank = %d). Exiting checkpoint callback.", self.rank)
            return

        if len(os.listdir(self.checkpoint_dir)) != 0:
            xgb_model, self.iteration = load_checkpoint(self.checkpoint_dir)
            current_iteration = self.iteration
        else:
            current_iteration = self.start_iteration + self.iteration
        self._save_checkpoint(model, current_iteration)

        # For example, if we are at iteration 5 and max_to_keep is 5, we no
        # longer need checkpoint from iteration 0 (i.e., xgboost-checkpoint.0),
        # so we put iteration_to_delete = 0 on the queue.
        iteration_to_delete = current_iteration - self.max_to_keep
        self.delete_queue.put(iteration_to_delete)

        offset_iteration = self.end_iteration if self.num_round is None else self.num_round

        training_has_ended = current_iteration + 1 >= self.start_iteration + offset_iteration

        if training_has_ended:
            self.stop()
        return False

    def start(self):
        """Start a background thread that deletes old checkpoints

        To delete stale checkpoints, we use a producer-consumer pattern: we
        start a daemon thread in the background and maintain a queue of files
        to delete.
        There may be a lock on the file if SageMaker is uploading the file; in
        that case, the file is put back on the queue and we try again later.
        When training is complete, we put SENTINEL on the queue, and when we
        see the SENTINEL, we clean up and exit the thread.
        """

        def _is_uploading(path):
            uploading = os.path.isfile(path + FILE_LOCK_SUFFIX)
            uploaded = os.path.isfile(path + FILE_SAFE_SUFFIX)
            return uploading and not uploaded

        def _should_skip(path):
            return not os.path.isfile(path) or path in self.previous_checkpoints

        def _remove(path):
            try:
                os.remove(path)
            except Exception:
                logger.debug("Failed to delete %s", path)
            finally:
                self.delete_queue.task_done()

        def _delete_uploaded_files():
            for iteration in iter(self.delete_queue.get, self.SENTINEL):
                path = self.format_path(iteration)
                if _should_skip(path):
                    self.delete_queue.task_done()
                    continue
                # If SageMaker is still uploading the file, we put the file back on the
                # queue and try again later. In order to avoid file corruption, we make
                # best attempt to not delete any files that are still being uploaded.
                if _is_uploading(path):
                    self.delete_queue.put(iteration)
                    continue
                _remove(path)
            self.delete_queue.task_done()

        def _cleanup():
            # Here, we've reached the end of training because we place sentinel in the
            # queue at the end of training. We put another sentinel, go through everything
            # in the queue once again, and try to remove it anyway whether there is a lock on
            # the file or not, because the training is done. On sagemaker, this should send
            # a delete signal to the agent so that the upload can be canceled and removed
            # from S3, if there are anything remaining in the queue. In normal cases,
            # _cleanup() exits almost immediately and does not do anything.
            self.delete_queue.put(self.SENTINEL)
            for iteration in iter(self.delete_queue.get, self.SENTINEL):
                path = self.format_path(iteration)
                _remove(path)
            self.delete_queue.task_done()

        def _delete_uploaded_files_and_cleanup():
            _delete_uploaded_files()
            _cleanup()

        self.thread = threading.Thread(target=_delete_uploaded_files_and_cleanup, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the background thread"""
        # placing a sentinel in the queue signals that training has ended.
        self.delete_queue.put(self.SENTINEL)
        # training has ended, so join the background back with the main thread.
        self.thread.join()

    def _save_checkpoint(self, model, iteration):
        """Save checkpoint to a file path formatted with iteration number"""
        with tempfile.NamedTemporaryFile(dir=self.checkpoint_dir, suffix=TEMP_FILE_SUFFIX, delete=False) as tf:
            model.save_model(tf.name)

        save_file_path = self.format_path(iteration)
        os.rename(tf.name, save_file_path)


def save_intermediate_model(intermediate_model_dir, model_name):
    """A callback function that saves intermediate models to disk.

    This is a wrapper function around SaveIntermediateModel.
    For details, see SaveIntermediateModel.
    """
    return SaveIntermediateModel(intermediate_model_dir=intermediate_model_dir, model_name=model_name)


class SaveIntermediateModel(object):
    """Create a callback that saves intermediate models to model directory on disk.

    The main purpose of this class is to support external early stopping techniques such as HPO,
    by saving intermediate copies of model directly to model_dir after each iteration of training.

    This is accomplished by overwriting the model after each iteration with 'model_name'.

    Attributes:
        intermediate_model_dir: path to the model directory where intermediate model
            will be saved.
        model_name: name of model

    Example:
        >>> save_intermediate_model = SaveIntermediateModel("/opt/ml/model", "xgboost-model")
        >>> xgboost.train(prams, dtrain, callbacks=[save_intermediate_model])
    """

    def __init__(self, intermediate_model_dir, model_name):
        """Init SaveIntermediateModel with intermediate_model_dir"""
        self.intermediate_model_dir = intermediate_model_dir
        self.model_name = model_name

        if not os.path.exists(self.intermediate_model_dir):
            os.makedirs(self.intermediate_model_dir)

    def __call__(self, env):
        """Make the class callable since it is meant be used as a callback"""
        return self.callback(env)

    def format_path(self):
        """Return a file path to intermediate model"""
        intermediate_path = os.path.join(self.intermediate_model_dir, self.model_name)
        return intermediate_path

    def save_intermediate_model(self, model):
        """Save intermediate model to intermediate model directory"""
        with tempfile.NamedTemporaryFile(dir=self.intermediate_model_dir, delete=False) as tf:
            model.save_model(tf.name)

        save_file_path = self.format_path()
        os.rename(tf.name, save_file_path)

    def callback(self, env):
        # env.rank: rabit rank of the node/process. master node has rank 0.
        # env.model: model object
        if env.rank != 0:
            logger.debug("Not master (rank = %d). Exiting intermediate model callback.", env.rank)
            return

        self.save_intermediate_model(env.model)


class SaveIntermediateModelCallBack(xgb.callback.TrainingCallback):
    """The new implementation of callback functions from 1.3."""

    def __init__(self, intermediate_model_dir, model_name, is_master):
        self.callback = SaveIntermediateModel(intermediate_model_dir, model_name)
        self.is_master = is_master

    def after_iteration(self, model, epoch, evals_log) -> bool:
        if self.is_master:
            self.callback.save_intermediate_model(model)
        return False
