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
from datetime import datetime
import os
import pprint
import unittest

from sagemaker_xgboost_container.algorithm_mode import channel_validation as cv
from sagemaker_xgboost_container.algorithm_mode import hyperparameter_validation as hpv
from sagemaker_xgboost_container.algorithm_mode import metadata
from sagemaker_xgboost_container.algorithm_mode import metrics as metrics_mod

import boto3


class TestCreateAlgorithm(unittest.TestCase):
    def test_create_algorithm(self):
        IMAGE_URI = os.getenv("TEST_IMAGE_URI")
        ALGORITHM_NAME = os.getenv("TEST_ALGORITHM_NAME")
        ROLE_ARN = os.getenv("TEST_ROLE_ARN")
        OUTPUT_PATH = os.getenv("TEST_OUTPUT_PATH")

        if IMAGE_URI is None:
            self.fail("Set TEST_IMAGE_URI environment variable.")
        if ALGORITHM_NAME is None:
            self.fail("Set TEST_ALGORITHM_NAME environment variable.")
        if ROLE_ARN is None:
            self.fail("Set TEST_ROLE_ARN environment variable.")
        if OUTPUT_PATH is None:
            self.fail("Set TEST_OUTPUT_PATH environment variable.")

        metrics = metrics_mod.initialize()
        hyperparameters = hpv.initialize(metrics)
        channels = cv.initialize()
        md = metadata.initialize(IMAGE_URI, hyperparameters, channels, metrics)

        client = boto3.client("sagemaker", region_name="us-west-2")
        try:
            client.delete_algorithm(AlgorithmName=ALGORITHM_NAME)
        except Exception as e:
            print(e)

        pprint.pprint(md)
        client.create_algorithm(AlgorithmName=ALGORITHM_NAME, **md)

        objective = metrics["validation:error"]
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d-%H%M%S")

        client.create_hyper_parameter_tuning_job(
            HyperParameterTuningJobName="test-hpo-" + dt_string,
            HyperParameterTuningJobConfig={
                "Strategy": "Random",
                "ResourceLimits": {
                    "MaxNumberOfTrainingJobs": 6,
                    "MaxParallelTrainingJobs": 2
                },
                "HyperParameterTuningJobObjective": objective.format_tunable(),
                "ParameterRanges": hyperparameters["alpha"].format_tunable_range()
            },
            TrainingJobDefinition={
                "AlgorithmSpecification": {
                    "AlgorithmName": ALGORITHM_NAME,
                    "TrainingInputMode": "File"
                },
                "StaticHyperParameters": {"num_round": "3"},
                "RoleArn": ROLE_ARN,
                "OutputDataConfig": {"S3OutputPath": OUTPUT_PATH},
                "ResourceConfig": {"InstanceType": "ml.m5.xlarge", "InstanceCount": 1, "VolumeSizeInGB": 5},
                "StoppingCondition": {"MaxRuntimeInSeconds": 300}
                }
        )
