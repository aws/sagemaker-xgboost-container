import os
import shutil
import pathlib
import tempfile
import time
import unittest
from unittest.mock import patch
from sagemaker_xgboost_container.checkpointing import SaveCheckpoint


class TestSaveCheckpoint(unittest.TestCase):

    def setUp(self):

        self.test_dir = tempfile.mkdtemp()

    @patch("xgboost.core.CallbackEnv")
    def test_not_master_node(self, env):

        callback = SaveCheckpoint(checkpoint_dir=self.test_dir)

        env.rank = 1
        env.iteration = 10
        callback(env)

        self.assertEqual(len(os.listdir(self.test_dir)), 0)

        callback.stop()

    @patch("xgboost.core.CallbackEnv")
    def test_SaveCheckpoint_single_iteration(self, env):

        env.rank = 0
        env.iteration = 42
        env.end_iteration = 100

        callback = SaveCheckpoint(checkpoint_dir=self.test_dir)
        callback(env)

        file_path = os.path.join(self.test_dir, "xgboost-checkpoint.42")
        self.assertTrue(os.path.isfile(file_path))
        self.assertTrue(len(os.listdir(self.test_dir)), 1)

        callback.stop()

    @patch("xgboost.core.CallbackEnv")
    def test_SaveCheckpoint_multiple_from_scratch(self, env):

        max_to_keep = 3
        end_iteration = 100

        env.rank = 0
        env.end_iteration = end_iteration

        callback = SaveCheckpoint(
            checkpoint_dir=self.test_dir,
            max_to_keep=max_to_keep)

        for i in range(end_iteration):
            env.iteration = i
            callback(env)

        expected_files = [
            "xgboost-checkpoint.97",
            "xgboost-checkpoint.98",
            "xgboost-checkpoint.99"]

        for fname in expected_files:
            fpath = os.path.join(self.test_dir, fname)
            self.assertTrue(os.path.isfile(fpath))

        self.assertTrue(len(os.listdir(self.test_dir)), 3)

        callback.stop()

    @patch("xgboost.core.CallbackEnv")
    def test_SaveCheckpoint_multiple_resume(self, env):

        max_to_keep = 3
        start_iteration = 10
        num_round = 10

        env.rank = 0

        callback = SaveCheckpoint(
            checkpoint_dir=self.test_dir,
            max_to_keep=max_to_keep,
            start_iteration=start_iteration,
            num_round=num_round)

        for i in range(num_round):
            env.iteration = i
            callback(env)

        expected_files = [
            "xgboost-checkpoint.17",
            "xgboost-checkpoint.18",
            "xgboost-checkpoint.19"]

        for fname in expected_files:
            fpath = os.path.join(self.test_dir, fname)
            self.assertTrue(os.path.isfile(fpath))

        self.assertTrue(len(os.listdir(self.test_dir)), 3)

    @patch("xgboost.core.CallbackEnv")
    def test_SaveCheckpoint_uploading(self, env):
        # This function attempts to simulate the file uploading procedure used
        # by SageMaker. When a file is being uploaded by SM, SM will create a
        # marker file (file name + .sagemaker-uploading) to indicate that the
        # file is being uploaded. SM will also create another marker file
        # (file name + .sagemaker-uploaded) when the upload is completed. Thus,
        # the background thread in SaveCheckpoint will skip deleting a file and
        # try again later if there is a marker file <filename>.sagemaker-uploading
        # and only attempt to delete a file when the marker file
        # <filename>.sagemaker-uploaded is present.
        max_to_keep = 1
        end_iteration = 100

        env.rank = 0
        env.end_iteration = end_iteration

        callback = SaveCheckpoint(
            checkpoint_dir=self.test_dir,
            max_to_keep=max_to_keep)

        env.iteration = 0
        callback(env)
        fpath = os.path.join(self.test_dir, "xgboost-checkpoint.0")
        self.assertTrue(os.path.isfile(fpath))

        fpath = os.path.join(self.test_dir, "xgboost-checkpoint.0")
        pathlib.Path(fpath + ".sagemaker-uploading").touch()

        env.iteration = 1
        callback(env)
        fpath = os.path.join(self.test_dir, "xgboost-checkpoint.1")
        self.assertTrue(os.path.isfile(fpath))

        fpath = os.path.join(self.test_dir, "xgboost-checkpoint.1")
        pathlib.Path(fpath + ".sagemaker-uploading").touch()

        env.iteration = 2
        callback(env)
        fpath = os.path.join(self.test_dir, "xgboost-checkpoint.2")
        self.assertTrue(os.path.isfile(fpath))

        fpath = os.path.join(self.test_dir, "xgboost-checkpoint.0")
        os.remove(fpath + ".sagemaker-uploading")
        time.sleep(0.5)
        self.assertFalse(os.path.isfile(fpath))

        env.iteration = 3
        callback(env)
        fpath = os.path.join(self.test_dir, "xgboost-checkpoint.3")
        self.assertTrue(os.path.isfile(fpath))

        env.iteration = 4
        callback(env)
        fpath = os.path.join(self.test_dir, "xgboost-checkpoint.4")
        self.assertTrue(os.path.isfile(fpath))

        self.assertFalse(os.path.isfile("xgboost-checkpoint.4"))

        callback.stop()

    def tearDown(self):

        shutil.rmtree(self.test_dir, ignore_errors=True)
