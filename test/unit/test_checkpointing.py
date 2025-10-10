import os
import logging
import pathlib
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import xgboost as xgb

from sagemaker_xgboost_container import checkpointing
from sagemaker_xgboost_container.checkpointing import SaveCheckpointCallBack, _sort_checkpoints


class TestSaveCheckpoint(unittest.TestCase):
    def setUp(self):

        self.test_dir = tempfile.mkdtemp()

    @patch("xgboost.core.Booster")
    def test_not_master_node(self, model):

        rank = 1
        iteration = 10

        callback = SaveCheckpointCallBack(checkpoint_dir=self.test_dir, rank=rank, iteration=iteration)

        callback(model)

        self.assertEqual(len(os.listdir(self.test_dir)), 0)

        callback.stop()

    @patch("xgboost.core.Booster")
    def test_SaveCheckpoint_single_iteration(self, model):

        rank = 0
        iteration = 42
        end_iteration = 100

        callback = SaveCheckpointCallBack(
            checkpoint_dir=self.test_dir, rank=rank, iteration=iteration, end_iteration=end_iteration
        )

        callback(model)

        file_path = os.path.join(self.test_dir, "xgboost-checkpoint.42")
        self.assertTrue(os.path.isfile(file_path))
        self.assertTrue(len(os.listdir(self.test_dir)), 1)

        callback.stop()

    @patch("xgboost.core.Booster")
    def test_SaveCheckpoint_multiple_from_scratch(self, model):

        rank = 0
        end_iteration = 100

        callback = SaveCheckpointCallBack(
            checkpoint_dir=self.test_dir, max_to_keep=3, rank=rank, end_iteration=end_iteration
        )

        for iteration in range(end_iteration):
            callback(model)

        expected_files = ["xgboost-checkpoint.97", "xgboost-checkpoint.98", "xgboost-checkpoint.99"]

        for fname in expected_files:
            fpath = os.path.join(self.test_dir, fname)
            self.assertTrue(os.path.isfile(fpath))

        self.assertTrue(len(os.listdir(self.test_dir)), 3)

        callback.stop()

    @patch("xgboost.core.Booster")
    def test_SaveCheckpoint_multiple_resume(self, model):

        num_round = 10

        rank = 0

        callback = SaveCheckpointCallBack(
            checkpoint_dir=self.test_dir, max_to_keep=3, start_iteration=10, num_round=10, rank=rank
        )

        for iteration in range(num_round):
            callback(model)

        expected_files = ["xgboost-checkpoint.17", "xgboost-checkpoint.18", "xgboost-checkpoint.19"]
        self.assertTrue(len(os.listdir(self.test_dir)), 3)
        assert sorted(os.listdir(self.test_dir)) == expected_files

        for fname in expected_files:
            fpath = os.path.join(self.test_dir, fname)
            self.assertTrue(os.path.isfile(fpath))

    @patch("xgboost.core.Booster")
    def test_SaveCheckpoint_uploading(self, model):
        # This function attempts to simulate the file uploading procedure used
        # by SageMaker. When a file is being uploaded by SM, SM will create a
        # marker file (file name + .sagemaker-uploading) to indicate that the
        # file is being uploaded. SM will also create another marker file
        # (file name + .sagemaker-uploaded) when the upload is completed. Thus,
        # the background thread in SaveCheckpoint will skip deleting a file and
        # try again later if there is a marker file <filename>.sagemaker-uploading
        # and only attempt to delete a file when the marker file
        # <filename>.sagemaker-uploaded is present.

        rank = 0
        end_iteration = 100

        callback = SaveCheckpointCallBack(
            checkpoint_dir=self.test_dir, max_to_keep=1, rank=rank, end_iteration=end_iteration
        )

        # For iteration 0
        callback(model)
        fpath = os.path.join(self.test_dir, "xgboost-checkpoint.0")
        self.assertTrue(os.path.isfile(fpath))

        fpath = os.path.join(self.test_dir, "xgboost-checkpoint.0")
        pathlib.Path(fpath + ".sagemaker-uploading").touch()

        # For iteration 1
        callback(model)
        fpath = os.path.join(self.test_dir, "xgboost-checkpoint.1")
        self.assertTrue(os.path.isfile(fpath))

        fpath = os.path.join(self.test_dir, "xgboost-checkpoint.1")
        pathlib.Path(fpath + ".sagemaker-uploading").touch()

        # For iteration 2
        callback(model)
        fpath = os.path.join(self.test_dir, "xgboost-checkpoint.2")
        self.assertTrue(os.path.isfile(fpath))

        fpath = os.path.join(self.test_dir, "xgboost-checkpoint.0")
        os.remove(fpath + ".sagemaker-uploading")
        time.sleep(0.5)
        self.assertFalse(os.path.isfile(fpath))

        # For iteration 3
        callback(model)
        fpath = os.path.join(self.test_dir, "xgboost-checkpoint.3")
        self.assertTrue(os.path.isfile(fpath))

        # For iteration 4
        callback(model)
        fpath = os.path.join(self.test_dir, "xgboost-checkpoint.4")
        self.assertTrue(os.path.isfile(fpath))

        self.assertFalse(os.path.isfile("xgboost-checkpoint.4"))

        callback.stop()

    def tearDown(self):

        shutil.rmtree(self.test_dir, ignore_errors=True)


def test_train(tmpdir, caplog):

    X_train = np.random.random(size=(100, 5))
    y_train = np.random.random(size=(100, 1))
    dtrain = xgb.DMatrix(X_train, label=y_train)

    X_test = np.random.random(size=(100, 5))
    y_test = np.random.random(size=(100, 1))
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {"objective": "binary:logistic"}

    train_args = dict(params=params, dtrain=dtrain, num_boost_round=20, evals=[(dtrain, "train"), (dtest, "test")])
    checkpoint_dir = os.path.join(tmpdir, "test_checkpoints")

    checkpointing.train(train_args, checkpoint_dir)

    # check that original train_args was not modified
    assert "callbacks" not in train_args
    assert "xgb_model" not in train_args
    assert "verbose_eval" not in train_args
    assert train_args["params"] == params
    assert train_args["dtrain"] is dtrain
    assert train_args["num_boost_round"] == 20
    # check that checkpoints were saved
    expected_files = [
        "xgboost-checkpoint.15",
        "xgboost-checkpoint.16",
        "xgboost-checkpoint.17",
        "xgboost-checkpoint.18",
        "xgboost-checkpoint.19",
    ]
    assert sorted(os.listdir(checkpoint_dir)) == expected_files

    train_args["num_boost_round"] = 30
    with caplog.at_level(logging.INFO):
        checkpointing.train(train_args, checkpoint_dir)

    assert "callbacks" not in train_args
    assert "xgb_model" not in train_args
    assert "verbose_eval" not in train_args
    assert train_args["num_boost_round"] == 30

    assert "Checkpoint loaded from" in caplog.text
    assert "Resuming from iteration 20" in caplog.text

    expected_files.extend(
        [
            "xgboost-checkpoint.25",
            "xgboost-checkpoint.26",
            "xgboost-checkpoint.27",
            "xgboost-checkpoint.28",
            "xgboost-checkpoint.29",
        ]
    )
    assert sorted(os.listdir(checkpoint_dir)) == expected_files


def test_train_zero_or_negative_rounds(tmpdir, caplog):

    X_train = np.random.random(size=(100, 5))
    y_train = np.random.random(size=(100, 1))
    dtrain = xgb.DMatrix(X_train, label=y_train)

    X_test = np.random.random(size=(100, 5))
    y_test = np.random.random(size=(100, 1))
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {"objective": "binary:logistic"}

    train_args = dict(params=params, dtrain=dtrain, num_boost_round=0, evals=[(dtrain, "train"), (dtest, "test")])
    checkpoint_dir = os.path.join(tmpdir, "test_checkpoints")

    bst = checkpointing.train(train_args, checkpoint_dir)
    assert isinstance(bst, xgb.Booster)
    assert not os.listdir(checkpoint_dir)

    train_args["num_boost_round"] = -1
    bst = checkpointing.train(train_args, checkpoint_dir)
    assert isinstance(bst, xgb.Booster)
    assert not os.listdir(checkpoint_dir)


def test_sort_checkpoints():
    input_checkpoints = [
        "xgboost-checkpoint.100",
        "xgboost-checkpoint.989",
        "xgboost-checkpoint.101",
        "xgboost-checkpoint.99",
    ]
    _sort_checkpoints(input_checkpoints)
    expected_sorted_checkpoints = [
        "xgboost-checkpoint.99",
        "xgboost-checkpoint.100",
        "xgboost-checkpoint.101",
        "xgboost-checkpoint.989",
    ]
    assert input_checkpoints == expected_sorted_checkpoints
