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
from sagemaker_algorithm_toolkit import metadata


def initialize(image_uri, hyperparameters, channels, metrics):
    training_spec = metadata.training_spec(
        hyperparameters, channels, metrics, image_uri, metadata.get_cpu_instance_types(metadata.Product.TRAINING), True
    )
    inference_spec = metadata.inference_spec(
        image_uri,
        metadata.get_cpu_instance_types(metadata.Product.HOSTING),
        metadata.get_cpu_instance_types(metadata.Product.BATCH_TRANSFORM),
        ["text/csv", "text/libsvm"],
        ["text/csv", "text/libsvm"],
    )
    return metadata.generate_metadata(training_spec, inference_spec)
