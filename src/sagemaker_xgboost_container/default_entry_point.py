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
import psutil
import os
import requests
import subprocess
import sys
import time
import traceback

# Copied from AI ALGS SDK
from sagemaker_xgboost_container.exceptions import convert_to_algorithm_error, PlatformError, CustomerError

from sagemaker_xgboost_container.train_helper import get_all_sizes, train_job, get_ip_from_host, start_yarn_daemons, \
    get_size, validate_file_format, submit_yarn_job, EASE_MEMORY, YARN_MASTER_MEMORY


ERROR_FILE = os.getenv("ALGO_ERROR_FILE")

INPUT_TRAIN_CONFIG_PATH = os.getenv("ALGO_INPUT_TRAINING_CONFIG_FILE")
INPUT_DATA_CONFIG_PATH = os.getenv("ALGO_INPUT_DATA_CONFIG_FILE")
INPUT_DATA_PATH = os.getenv("ALGO_INPUT_DATA_DIR")
INPUT_RESOURCE_CONFIG_PATH = os.getenv("ALGO_INPUT_RESOURCE_CONFIG_FILE")

MODEL_DIR = os.getenv("ALGO_MODEL_DIR")

OUTPUT_DIR = os.getenv("ALGO_OUTPUT_DIR")
OUTPUT_FAILED_FILE = os.getenv("ALGO_OUTPUT_FAILED_FILE")


INPUT_DATA_PATH = os.getenv("ALGO_INPUT_DATA_DIR")
HTTP_SERVER_PORT = "8000"


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
            distributed_train(resource_config, train_config, data_config)

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


def distributed_train(resource_config, train_config, data_config):
    current_host = resource_config["current_host"]
    master_host = min(resource_config["hosts"])
    num_hosts = len(resource_config["hosts"])

    if num_hosts == 1:
        raise PlatformError("Distributed training should run with more than one host")

    master_ip = get_ip_from_host(master_host)
    host_ip = get_ip_from_host(current_host)

    logging.info(
        "Number of hosts: {}, master IP address: {}, host IP address: {}.".format(num_hosts, master_ip, host_ip))
    start_yarn_daemons(num_hosts, current_host, master_host, master_ip)

    # adding duplicated validation code before submit yarn job
    channels = list(data_config.keys())
    file_type = data_config[channels[0]].get("ContentType", "libsvm")
    if 'train' not in channels:
        raise CustomerError("Channelname train is required for training")

    TRAIN_CHANNEL = 'train'
    VAL_CHANNEL = 'validation'
    train_path = INPUT_DATA_PATH + '/' + TRAIN_CHANNEL
    val_path = INPUT_DATA_PATH + '/' + VAL_CHANNEL

    s3_dist_type_train = data_config[TRAIN_CHANNEL].get("S3DistributionType")
    s3_dist_type_val = data_config[VAL_CHANNEL].get("S3DistributionType") if data_config.get(
        VAL_CHANNEL) is not None else None

    train_files_size = get_size(train_path)
    val_files_size = get_size(val_path)
    real_train_mem_size = train_files_size / num_hosts if s3_dist_type_train == "FullyReplicated" else train_files_size
    real_val_mem_size = val_files_size / num_hosts if s3_dist_type_val == "FullyReplicated" else val_files_size

    mem_size = psutil.virtual_memory().available
    real_mem_size = mem_size - EASE_MEMORY - YARN_MASTER_MEMORY

    logging.info("File size need to be processed in the node: {}. Available memory size in the node: {}"
                 .format(str(round((real_train_mem_size + real_val_mem_size) / (1024 * 1024), 2)) + 'mb',
                         str(round(real_mem_size / (1024 * 1024), 2)) + 'mb'))
    if real_mem_size <= 0:
        raise CustomerError(
            "Insufficient memory available to run distributed training. Please switch to larger instance with more RAM.")

    exceed_memory = (real_train_mem_size + real_val_mem_size) > real_mem_size
    if exceed_memory and file_type.lower() == 'csv':
        raise CustomerError("Insufficient memory available to run distributed training with given data. CSV input does \
                            not support external memory. Switch to large instance with more RAM or run with more instances.")
    elif exceed_memory and file_type.lower() == 'libsvm':
        logging.info("Insufficient memory available to run distributed training with given data. Using external memory for libsvm input.\
                       Switch to larger instance with more RAM or run with more instances for better performance and lower cost.")

    # validate file format
    validate_file_format(train_path, file_type)
    validate_file_format(val_path, file_type)

    if current_host == master_host:
        FNULL = open(os.devnull, 'w')
        server = subprocess.Popen(["python3", "-m", "SimpleHTTPServer", HTTP_SERVER_PORT], stdout=FNULL, stderr=FNULL)

        if not server.poll():
            logging.info("HTTP server started....")
        else:
            raise PlatformError("Could not start HTTP server. Terminating!")

        submit_yarn_job(train_config, host_ip, num_hosts)

        server.terminate()  # End HTTP server
    else:
        # alive check for master node, if master node exit, all slave nodes exit
        time.sleep(30)
        while master_ip != '':
            try:
                r = requests.get('http://%s:%s' % (master_ip, HTTP_SERVER_PORT))
                status_code = r.status_code
                if status_code != 200:
                    raise requests.exceptions.ConnectionError
                time.sleep(15)
            except requests.exceptions.ConnectionError as e:
                master_ip = ""

        logging.info("Master host is not alive. Training might have finished. Shutting down.... Check the logs for "
                     "algo-1 machine.")