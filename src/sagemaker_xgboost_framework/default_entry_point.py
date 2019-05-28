# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import json
import logging
import os
import sys
import traceback

# Copied from AI ALGS SDK
from sagemaker_xgboost_framework.exceptions import convert_to_algorithm_error, PlatformError, CustomerError

from sagemaker_xgboost_framework.train_helper import get_all_sizes, train_job


ERROR_FILE = os.getenv("ALGO_ERROR_FILE")

INPUT_TRAIN_CONFIG_PATH = os.getenv("ALGO_INPUT_TRAINING_CONFIG_FILE")
INPUT_DATA_CONFIG_PATH = os.getenv("ALGO_INPUT_DATA_CONFIG_FILE")
INPUT_DATA_PATH = os.getenv("ALGO_INPUT_DATA_DIR")
INPUT_RESOURCE_CONFIG_PATH = os.getenv("ALGO_INPUT_RESOURCE_CONFIG_FILE")

MODEL_DIR = os.getenv("ALGO_MODEL_DIR")

OUTPUT_DIR = os.getenv("ALGO_OUTPUT_DIR")
OUTPUT_FAILED_FILE = os.getenv("ALGO_OUTPUT_FAILED_FILE")


def algorithm_mode_train():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    try:

        with open(INPUT_RESOURCE_CONFIG_PATH, "r") as f:
            resource_config = json.load(f)
        with open(INPUT_TRAIN_CONFIG_PATH, "r") as f:
            train_config = json.load(f)
        with open(INPUT_DATA_CONFIG_PATH, "r") as f:
            data_config = json.load(f)

        num_hosts = len(resource_config["hosts"])
        if num_hosts < 1:
            raise PlatformError("Number of hosts should be greater or equal to 1")

        if num_hosts == 1:
            logging.info("Running standalone xgboost training.")
            train_job(resource_config, train_config, data_config)
        else:
            raise CustomerError("Running distributed xgboost training; this is not supported yet.")

    except Exception as e:
        with open(OUTPUT_FAILED_FILE, "w") as f:
            f.write(convert_to_algorithm_error(e).failure_message())
        logging.exception(convert_to_algorithm_error(e).failure_message())

        train_files_size, val_files_size, mem_size = get_all_sizes()

        with open(ERROR_FILE, "w") as f:
            f.write("Stacktrace for debugging:\n Dataset "
                    "size:\nTraining:{}\nValidation:{}\nMemory size:{}\n{}\n{}\n".format(
                        train_files_size, val_files_size, mem_size, type(e), e))
            f.write("Traceback:\n")
            traceback.print_exc(file=f)

        sys.exit(1)
