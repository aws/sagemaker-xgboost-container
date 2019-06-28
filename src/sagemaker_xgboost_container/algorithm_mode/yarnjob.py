import json
import logging
import os
import sys
import traceback

import xgboost as xgb

from sagemaker_xgboost_container.algorithm_mode import channel_validation as cv
from sagemaker_xgboost_container.algorithm_mode import hyperparameter_validation as hpv
from sagemaker_xgboost_container.algorithm_mode import metrics as metrics_mod

from sagemaker_xgboost_container.algorithm_mode import signals
from sagemaker_xgboost_container.algorithm_mode.train_helper import train_job, get_all_sizes
from sagemaker_algorithm_toolkit.exceptions import convert_to_algorithm_error

INPUT_TRAIN_CONFIG_PATH = os.getenv("ALGO_INPUT_TRAINING_CONFIG_FILE")
INPUT_DATA_CONFIG_PATH = os.getenv("ALGO_INPUT_DATA_CONFIG_FILE")
INPUT_RESOURCE_CONFIG_PATH = os.getenv("ALGO_INPUT_RESOURCE_CONFIG_FILE")
OUTPUT_FAILED_FILE = os.getenv("ALGO_OUTPUT_FAILED_FILE")
ERROR_FILE = os.getenv("ALGO_ERROR_FILE")

xgb.rabit.init()
logging.info("DOHYEONG starting rabit")

try:
    # Install termiante/signal handlers
    # signals.install_terminate_and_signal_handlers()

    with open(INPUT_RESOURCE_CONFIG_PATH, "r") as f:
        resource_config = json.load(f)
    with open(INPUT_TRAIN_CONFIG_PATH, "r") as f:
        train_config = json.load(f)
    with open(INPUT_DATA_CONFIG_PATH, "r") as f:
        data_config = json.load(f)

    metrics = metrics_mod.initialize()

    hyperparameters = hpv.initialize(metrics)
    final_train_config = hyperparameters.validate(train_config)

    channels = cv.initialize()
    final_data_config = channels.validate(data_config)

    train_job(resource_config, final_train_config, final_data_config)
except Exception as e:
    with open(OUTPUT_FAILED_FILE, "w") as f:
        f.write(convert_to_algorithm_error(e).failure_message())
    logging.exception(convert_to_algorithm_error(e).failure_message())

    train_files_size, val_files_size, mem_size = get_all_sizes()
    worker_id = xgb.rabit.get_rank()
    with open(ERROR_FILE, "w") as f:
        f.write("Worker#{}:\n".format(worker_id))
        f.write("Stacktrace for debugging:\n Dataset "
                "size:\nTraining:{}\nValidation:{}\nMemory size:{}\n{}\n{}\n"
                .format(train_files_size, val_files_size, mem_size, type(e), e))
        f.write("Traceback:\n")
        traceback.print_exc(file=f)

    sys.exit(1)

xgb.rabit.tracker_print("Finished training\n")
xgb.rabit.finalize()
