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
import os
from sagemaker_containers.beta.framework import env
from sagemaker_xgboost_container.algorithm_mode import serve


# Pre-load the model in the algorithm mode.
# Otherwise, the model will be loaded when serving the first request per worker.
# When the model is large, the request may timeout.
if os.environ.get("SERVER_SOFTWARE") is not None and env.ServingEnv().module_name is None:
    serve.ScoringService.load_model()
