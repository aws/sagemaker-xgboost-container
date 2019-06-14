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
from sagemaker_algorithm_toolkit import exceptions as exc

import logging


class Metric(object):
    MAXIMIZE = "Maximize"
    MINIMIZE = "Minimize"

    def __init__(self, name, regex, format_string=None, tunable=True, direction=None):
        self.name = name
        self.format_string = format_string
        self.direction = direction
        self.regex = regex
        self.tunable = tunable
        if self.tunable and direction is None:
            raise exc.AlgorithmError("direction must be specified if tunable is True.")

    def log(self, value):
        logging.info(self.format_string.format(value))

    def format_tunable(self):
        return {"MetricName": self.name,
                "Type": self.direction}

    def format_definition(self):
        return {"Name": self.name,
                "Regex": self.regex}


class Metrics(object):
    def __init__(self, *metrics):
        self.metrics = {metric.name: metric for metric in metrics}

    def __getitem__(self, name):
        return self.metrics[name]

    @property
    def names(self):
        return list(self.metrics)

    def format_tunable(self):
        metrics = []
        for name, metric in self.metrics.items():
            if metric.tunable:
                metrics.append(metric.format_tunable())
        return metrics

    def format_definitions(self):
        return [metric.format_definition() for name, metric in self.metrics.items()]
