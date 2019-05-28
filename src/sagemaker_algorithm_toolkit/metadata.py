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

import boto3


def _get_instance_types(region_name="us-east-1", location="US East (N. Virginia)"):
    s = boto3.client("pricing", region_name=region_name)

    NAME = 'AmazonSageMaker'
    FILTERS = [
        {"Type": "TERM_MATCH", "Field": "productFamily", "Value": "ML Instance"},
        {"Type": "TERM_MATCH", "Field": "location", "Value": location}]
    results = s.get_products(ServiceCode=NAME, Filters=FILTERS)

    total_results = []
    while results.get("NextToken"):
        total_results += results["PriceList"]
        results = s.get_products(ServiceCode=NAME, Filters=FILTERS, NextToken=results["NextToken"])

    instance_types = {}
    for result in total_results:
        result = json.loads(result)
        instance_type = result["product"]["attributes"]["instanceType"]
        gpu = result["product"]["attributes"]["gpu"]

        instance_types[instance_type] = int(gpu)
    return instance_types


class Product:
    NOTEBOOK = "Notebook"
    TRAINING = "Training"
    HOSTING = "Hosting"
    BATCH_TRANSFORM = "BatchTransform"


def _trim(instance_type_product):
    SEPARATOR = "-"  # e.g. ml.p3.2xlarge-Hosting
    return instance_type_product.split(SEPARATOR)[0]


def get_cpu_instance_types(product, **kwargs):
    results = []
    for instance_type, gpu_count in _get_instance_types(**kwargs).items():
        if gpu_count == 0 and product in instance_type:
            results.append(_trim(instance_type))
    return results


def get_single_gpu_instance_types(product, **kwargs):
    results = []
    for instance_type, gpu_count in _get_instance_types(**kwargs).items():
        if gpu_count == 1 and product in instance_type:
            results.append(_trim(instance_type))
    return results


def get_multi_gpu_instance_types(product, **kwargs):
    results = []
    for instance_type, gpu_count in _get_instance_types(**kwargs).items():
        if gpu_count > 1 and product in instance_type:
            results.append(_trim(instance_type))
    return results


def training_spec(hyperparameters, channels, metrics,
                  image_uri,
                  supported_training_instance_types,
                  supports_distributed_training):
    return {
        "TrainingImage": image_uri,
        "TrainingChannels": channels.format(),
        "SupportedHyperParameters": hyperparameters.format(),
        "SupportedTrainingInstanceTypes": supported_training_instance_types,
        "SupportsDistributedTraining": supports_distributed_training,
        "MetricDefinitions": metrics.format_definitions(),
        "SupportedTuningJobObjectiveMetrics": metrics.format_tunable(),
        }


def inference_spec(image_uri,
                   supported_realtime_inference_instance_types,
                   supported_transform_inference_instance_types,
                   supported_content_types,
                   supported_response_mimetypes):
    return {
        "Containers": [{"Image": image_uri}],
        "SupportedTransformInstanceTypes": supported_transform_inference_instance_types,
        "SupportedRealtimeInferenceInstanceTypes": supported_realtime_inference_instance_types,
        "SupportedContentTypes": supported_content_types,
        "SupportedResponseMIMETypes": supported_response_mimetypes
        }


def generate_metadata(training_spec, inference_spec):
    return {"TrainingSpecification": training_spec,
            "InferenceSpecification": inference_spec}
