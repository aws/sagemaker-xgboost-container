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
import psutil
import requests
import subprocess
import time

from sagemaker_algorithm_toolkit import exceptions
from sagemaker_xgboost_container.algorithm_mode import channel_validation as cv
from sagemaker_xgboost_container.algorithm_mode import hyperparameter_validation as hpv
from sagemaker_xgboost_container.algorithm_mode import metrics as metrics_mod
from sagemaker_xgboost_container.algorithm_mode.train_helper import train_job, get_ip_from_host, \
    start_yarn_daemons, submit_yarn_job, get_size, validate_file_format, EASE_MEMORY, YARN_MASTER_MEMORY


INPUT_DATA_PATH = os.getenv("ALGO_INPUT_DATA_DIR")
HTTP_SERVER_PORT = "8000"


def algorithm_mode_train():
    # TODO: replace with CSDK constants in sagemaker_containers._env
    INPUT_TRAIN_CONFIG_PATH = os.getenv("ALGO_INPUT_TRAINING_CONFIG_FILE")
    INPUT_DATA_CONFIG_PATH = os.getenv("ALGO_INPUT_DATA_CONFIG_FILE")
    INPUT_RESOURCE_CONFIG_PATH = os.getenv("ALGO_INPUT_RESOURCE_CONFIG_FILE")
    # END TODO

    resource_config = json.load(open(INPUT_RESOURCE_CONFIG_PATH, "r"))
    train_config = json.load(open(INPUT_TRAIN_CONFIG_PATH, "r"))
    data_config = json.load(open(INPUT_DATA_CONFIG_PATH, "r"))

    num_hosts = len(resource_config["hosts"])
    if num_hosts < 1:
        raise exceptions.PlatformError("Number of hosts should be greater or equal to 1")

    metrics = metrics_mod.initialize()

    hyperparameters = hpv.initialize(metrics)
    final_train_config = hyperparameters.validate(train_config)

    channels = cv.initialize()
    final_data_config = channels.validate(data_config)

    logging.info("hyperparameters {}".format(final_train_config))
    logging.info("channels {}".format(final_data_config))

    if num_hosts > 1:
        distributed_train(resource_config, final_train_config, final_data_config)
    else:
        train_job(resource_config, final_train_config, final_data_config)


def distributed_train(resource_config, train_config, data_config):
    current_host = resource_config["current_host"]
    master_host = min(resource_config["hosts"])
    num_hosts = len(resource_config["hosts"])

    if num_hosts == 1:
        raise exceptions.PlatformError("Distributed training should run with more than one host")

    master_ip = get_ip_from_host(master_host)
    logging.info("Master IP: {}".format(master_ip))
    host_ip = get_ip_from_host(current_host)
    logging.info("Currrent IP: {}".format(host_ip))

    logging.info(
        "Number of hosts: {}, master IP address: {}, host IP address: {}.".format(num_hosts, master_ip, host_ip))
    start_yarn_daemons(num_hosts, current_host, master_host, master_ip)

    # adding duplicated validation code before submit yarn job
    channels = list(data_config.keys())
    file_type = data_config[channels[0]].get("ContentType", "libsvm")
    if 'train' not in channels:
        raise exceptions.UserError("Channelname train is required for training")

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
        raise exceptions.UserError("Insufficient memory available to run distributed training. "
                                   "Please switch to larger instance with more RAM.")

    exceed_memory = (real_train_mem_size + real_val_mem_size) > real_mem_size
    if exceed_memory and file_type.lower() == 'csv':
        raise exceptions.UserError("Insufficient memory available to run distributed training with given data. "
                                   "CSV input does not support external memory. "
                                   "Switch to large instance with more RAM or run with more instances.")
    elif exceed_memory and file_type.lower() == 'libsvm':
        logging.info("Insufficient memory available to run distributed training with given data. "
                     "Using external memory for libsvm input. "
                     "Switch to larger instance with more RAM or run with more instances "
                     "for better performance and lower cost.")

    # validate file format
    validate_file_format(train_path, file_type)
    validate_file_format(val_path, file_type)

    if current_host == master_host:
        FNULL = open(os.devnull, 'w')
        server_cmd = "python3 -m SimpleHTTPServer {}".format(HTTP_SERVER_PORT)
        logging.info("Starting server command: {}".format(server_cmd))
        server = subprocess.Popen(["python3", "-m", "SimpleHTTPServer", HTTP_SERVER_PORT], stdout=FNULL, stderr=FNULL)

        if not server.poll():
            logging.info("HTTP server started....")
        else:
            raise exceptions.PlatformError("Could not start HTTP server. Terminating!")

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
