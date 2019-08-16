import os
import shutil
import pathlib
import tempfile
import time
import unittest
from unittest.mock import patch
from sagemaker_xgboost_container.algorithm_mode.callback import SaveCheckpoint


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

        file_path = os.path.join(self.test_dir, "xgboost-checkpoint.000000000042")
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
            "xgboost-checkpoint.000000000097",
            "xgboost-checkpoint.000000000098",
            "xgboost-checkpoint.000000000099"]

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
            "xgboost-checkpoint.000000000017",
            "xgboost-checkpoint.000000000018",
            "xgboost-checkpoint.000000000019"]

        for fname in expected_files:
            fpath = os.path.join(self.test_dir, fname)
            self.assertTrue(os.path.isfile(fpath))

        self.assertTrue(len(os.listdir(self.test_dir)), 3)

    @patch("xgboost.core.CallbackEnv")
    def test_SaveCheckpoint_uploading(self, env):

        max_to_keep = 1
        end_iteration = 100

        env.rank = 0
        env.end_iteration = end_iteration

        callback = SaveCheckpoint(
            checkpoint_dir=self.test_dir,
            max_to_keep=max_to_keep)

        env.iteration = 0
        callback(env)
        fpath = os.path.join(self.test_dir, "xgboost-checkpoint.000000000000")
        self.assertTrue(os.path.isfile(fpath))

        fpath = os.path.join(self.test_dir, "xgboost-checkpoint.000000000000")
        pathlib.Path(fpath + ".sagemaker-uploading").touch()

        env.iteration = 1
        callback(env)
        fpath = os.path.join(self.test_dir, "xgboost-checkpoint.000000000001")
        self.assertTrue(os.path.isfile(fpath))

        fpath = os.path.join(self.test_dir, "xgboost-checkpoint.000000000001")
        pathlib.Path(fpath + ".sagemaker-uploading").touch()

        env.iteration = 2
        callback(env)
        fpath = os.path.join(self.test_dir, "xgboost-checkpoint.000000000002")
        self.assertTrue(os.path.isfile(fpath))

        fpath = os.path.join(self.test_dir, "xgboost-checkpoint.000000000000")
        os.remove(fpath + ".sagemaker-uploading")
        time.sleep(0.5)
        self.assertFalse(os.path.isfile(fpath))

        env.iteration = 3
        callback(env)
        fpath = os.path.join(self.test_dir, "xgboost-checkpoint.000000000003")
        self.assertTrue(os.path.isfile(fpath))

        env.iteration = 4
        callback(env)
        fpath = os.path.join(self.test_dir, "xgboost-checkpoint.000000000004")
        self.assertTrue(os.path.isfile(fpath))

        self.assertFalse(os.path.isfile("xgboost-checkpoint.000000000004"))

        callback.stop()

    def tearDown(self):

        shutil.rmtree(self.test_dir, ignore_errors=True)
