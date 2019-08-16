import logging
import os
import tempfile
import threading
import queue
from typing import Callable, Optional
from xgboost import rabit
from xgboost.core import CallbackEnv
from xgboost.callback import _fmt_metric as fmt_metric


TEMP_FILE_SUFFIX = ".sagemaker-ignore"
FILE_LOCK_SUFFIX = ".sagemaker-uploading"
FILE_SAFE_SUFFIX = ".sagemaker-uploaded"

CHECKPOINT_FILENAME = "xgboost-checkpoint"
CHECKPOINT_NUM_DIGITS = 12

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _zero_pad(iteration: int) -> str:

    iter_num = format(iteration, "0{}".format(CHECKPOINT_NUM_DIGITS))

    return iter_num


# modified from https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/callback.py
def print_evaluation(period=1, show_stdv=True, start_iteration=0):
    """Create a callback that print evaluation result.

    We print the evaluation results every **period** iterations
    and on the first and the last iterations.

    Parameters
    ----------
    period : int
        The period to log the evaluation results

    show_stdv : bool, optional
         Whether show stdv if provided

    Returns
    -------
    callback : function
        A callback that print evaluation every period iterations.
    """
    def callback(env):
        """internal function"""
        if env.rank != 0 or (not env.evaluation_result_list) or period is False or period == 0:
            return
        i = env.iteration
        if i % period == 0 or i + 1 == env.begin_iteration or i + 1 == env.end_iteration:
            msg = '\t'.join([fmt_metric(x, show_stdv) for x in env.evaluation_result_list])
            rabit.tracker_print('[%d]\t%s\n' % (i + start_iteration, msg))
    return callback


class SaveCheckpoint:
    """Create a callback that saves checkpoints to disk.

    This class represents the hook which is meant to be used a callback
    function in XGBoost.

    Example
    -------
    >>> save_checkpoint = SaveCheckpoint("/opt/ml/checkpoints")
    >>> xgboost.train(prams, dtrain, callbacks=[save_checkpoint])

    At the end of each iteration, we save each checkpoint to different files
    parametrized by the iteration number, e.g., checkpoint.01, checkpoint.02,
    etc. However, this could result in a large number of files to save locally
    (and on S3 if used in SageMaker). To save disk space and reduce the amount
    of data required to be downloaded on resumption, we retain only the N
    (default 5, spcified by 'max_to_keep') most recent checkpoints.

    To delete stale checkpoints, we use a producer-consumer pattern: we start a
    daemon thread in the background and maintain a queue of files to delete.
    There may be a lock on the file if SageMaker is uploading the file; in that
    case, the file is put back on the queue and we try again later. When
    training is complete, we put SENTINEL on the queue, and when we see the
    SENTINEL, we clean up and exit the thread.
    """

    SENTINEL = None

    def __init__(
            self, checkpoint_dir: str, max_to_keep: int = 5, start_iteration: int = 0,
            num_round: Optional[int] = None
            ) -> None:
        """
        Parameters
        ----------
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
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        self.start_iteration = start_iteration
        self.num_round = num_round

        self.step = start_iteration
        self.previous_checkpoints = [
            os.path.join(self.checkpoint_dir, f)
            for f in os.listdir(self.checkpoint_dir)]

        self.thread = None
        self.delete_queue = queue.Queue()

        self.start()

    def __call__(self, env: CallbackEnv) -> Callable[[CallbackEnv], None]:

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        return self.callback(env)

    def fmt_path(self, i: int) -> str:

        filename = "{}.{}".format(CHECKPOINT_FILENAME, _zero_pad(i))
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        return checkpoint_path

    def start(self) -> None:

        def _delete_once(skip_locked_files=True):

            for i in iter(self.delete_queue.get, self.SENTINEL):

                path = self.fmt_path(i)
                if (skip_locked_files
                        and os.path.isfile(path + FILE_LOCK_SUFFIX)
                        and not os.path.isfile(path + FILE_SAFE_SUFFIX)):
                    self.delete_queue.put(i)
                    continue
                try:
                    os.remove(path)
                except Exception:
                    logger.debug("Failed to delete %s", path)
                finally:
                    self.delete_queue.task_done()

            self.delete_queue.task_done()

        def _delete_twice():

            _delete_once(skip_locked_files=True)
            # Here, we've reached the end of training because we place sentinel in the
            # queue at the end of training. We put another sentinel, go through everything
            # in the queue once again, and try to remove it anyway where there is a lock on
            # the file or not, because the training is done. On sagemaker, this should send
            # a delete signal to the agent so that the upload can be canceled and removed
            # from S3.
            self.delete_queue.put(self.SENTINEL)
            _delete_once(skip_locked_files=False)

        self.thread = threading.Thread(target=_delete_twice, daemon=True)
        self.thread.start()

    def stop(self) -> None:

        self.delete_queue.put(self.SENTINEL)
        self.thread.join()

    def callback(self, env: CallbackEnv) -> None:

        # env.rank: rabit rank of the node/process. master node has rank 0.
        # env.iteration: current boosting round
        # env.begin_iteration: round # when training started. this is always 0.
        # env.end_iteration: round # when training will end. this is always num_round + 1.
        # env.model: model object

        if env.rank != 0:
            logger.debug("Not master (rank = %d). Exiting checkpoint callback.", env.rank)
            return

        i = self.start_iteration + env.iteration

        with tempfile.NamedTemporaryFile(
                dir=self.checkpoint_dir, suffix=TEMP_FILE_SUFFIX, delete=False
                ) as tf:
            env.model.save_model(tf.name)

        os.rename(tf.name, self.fmt_path(i))

        target_file = self.fmt_path(i - self.max_to_keep)
        if (os.path.isfile(target_file)
                and target_file not in self.previous_checkpoints):
            self.delete_queue.put(i - self.max_to_keep)

        if ((self.num_round is not None and i + 1 >= self.start_iteration + self.num_round)
                or (self.num_round is None and i + 1 >= self.start_iteration + env.end_iteration)):
            self.stop()
