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
from sagemaker_xgboost_framework.exceptions import AlgorithmError
from sagemaker_xgboost_framework.metrics.utils import Preconditions

LOSS_METRIC_TYPE = 'loss'
SCORE_METRIC_TYPE = 'score'
THROUGHPUT_METRIC_TYPE = 'throughput'
PROGRESS_METRIC_TYPE = 'progress'

FREQUENCY_FINAL = 'final'
FREQUENCY_PER_EPOCH = 'epoch'
FREQUENCY_PER_BATCH = 'batch'
FREQUENCY_PER_CHECKPOINT = 'checkpoint'
ALLOWED_FREQUENCIES = [FREQUENCY_FINAL, FREQUENCY_PER_EPOCH, FREQUENCY_PER_BATCH, FREQUENCY_PER_CHECKPOINT]


def get_metric_emission_frequency(epoch, batch, checkpoint):
    """Determine metric emission frequency based on the set of parameters.

    Metric is emitted once if all parameters are None (only epoch value is checked).
    Metric is emitted once per epoch if only epoch has a non-None value.
    Metric is emitted once ber batch if epoch and batch have non-None values.
    Metric is emitted once per checkpoint if epoch and checkpoint are not None.

    :param epoch: epoch number
    :param batch: batch number
    :param checkpoint: checkpoint number
    :return: emission frequency
    """
    if epoch is None:
        return FREQUENCY_FINAL

    if batch is not None:
        return FREQUENCY_PER_BATCH

    if checkpoint is not None:
        return FREQUENCY_PER_CHECKPOINT

    return FREQUENCY_PER_EPOCH


class PerformanceMetricDefinition(object):
    """
    Base class representing performance metric.
    """
    def __init__(self, name, m_type, emission_frequencies,
                 tunable, default_frequency=None, name_regex=None, description=None):
        """
        initializes the common parameters for a metric
        :param name: name of the metric.
        :type name: string
        :param m_type: type of the metric.
        :type m_type: string
        :param emission_frequencies: the frequencies at which the metric is emitted.
        :type emission_frequencies: set of possible frequencies.
        :param tunable: True, if the metric is tunable, False otherwise
        :type tunable: bool
        :param default_frequency: frequency at which the metrics is emitted
        :type default_frequency: str or None
        :param name_regex: for parameterized metrics the name may not be static. name_regex should
                           be a regex string that would match the 'runtime' name of the metric.
        :type name_regex: string
        :param description: description of the metric (to be used for docs)
        :type description: string
        """

        self.name = Preconditions.check_not_none(name)
        self.name_regex = name_regex or name
        self.type = Preconditions.check_not_none(m_type)
        self.emission_frequencies = Preconditions.check_not_none(emission_frequencies)
        self.tunable = Preconditions.check_allowed_value(tunable, [True, False])
        self.description = description
        if not default_frequency and len(emission_frequencies) > 1:
            raise AlgorithmError("Default emission frequency should be set, when number of possible "
                                 "emission frequencies {} is more than 1".format(emission_frequencies))
        self.default_frequency = default_frequency or list(emission_frequencies)[0]

    def __eq__(self, other):
        return (self.name == other.name) and \
               (self.name_regex == other.name_regex) and \
               (self.type == other.type) and \
               (self.emission_frequencies == other.emission_frequencies) and \
               (self.default_frequency == other.default_frequency) and \
               (self.tunable == other.tunable) and \
               (self.description == other.description)


class QualityMetricDefinition(PerformanceMetricDefinition):
    def __init__(self, name, m_type, emission_frequencies,
                 tunable, default_frequency=None, name_regex=None, description=None):
        super(QualityMetricDefinition, self).__init__(name=name, m_type=m_type,
                                                      emission_frequencies=emission_frequencies,
                                                      default_frequency=default_frequency,
                                                      tunable=tunable, name_regex=name_regex, description=description)
        Preconditions.check_allowed_value(m_type, [LOSS_METRIC_TYPE, SCORE_METRIC_TYPE])


class ThroughputMetricDefinition(PerformanceMetricDefinition):
    """
    This class represents throughput metric definition.
    """
    def __init__(self, name, emission_frequencies, default_frequency=None,
                 tunable=False, name_regex=None, description=None, unit='records/second'):
        """
        :param name: name of the metric.
        :type name: string
        :param emission_frequencies: the frequencies at which the metric is emitted.
        :type emission_frequencies: set of possible frequencies.
        :param default_frequency: the frequency at which the metric is emitted by default
        :type default_frequency: string
        :param tunable: True, if the metric is tunable, False otherwise. Defaults to  False.
        :type tunable: bool
        :param name_regex: for parameterized metrics the name may not be static. name_regex should
                           be a regex string that would match the 'runtime' name of the metric.
        :type name_regex: string
        :param description: description of the metric (to be used for docs)
        :type description: string
        :param unit: unit in which the throughput is reported. Defaults to 'records/second'
        :type unit: string
        """

        super(ThroughputMetricDefinition, self).__init__(
            name=name, m_type=THROUGHPUT_METRIC_TYPE,
            emission_frequencies=emission_frequencies,
            default_frequency=default_frequency, tunable=tunable,
            name_regex=name_regex, description=description)
        Preconditions.check_allowed_value(self.type, [THROUGHPUT_METRIC_TYPE])
        self.unit = Preconditions.check_not_none(unit)

    def __eq__(self, other):
        ret_val = super(ThroughputMetricDefinition, self).__eq__(other=other)
        return ret_val and self.unit == other.unit


class ProgressMetricDefinition(PerformanceMetricDefinition):
    """
    This class represent progress metric definition.
    """
    def __init__(self, name, emission_frequencies,
                 default_frequency=None, name_regex=None, description=None, unit='%'):
        """
        :param name: name of the metric.
        :type name: string
        :param emission_frequencies: the frequencies at which the metric is emitted.
        :type emission_frequencies: set of possible frequencies.
        :param default_frequency: the frequency at which the metric is emitted by default
        :type default_frequency: string
        :param name_regex: for parameterized metrics the name may not be static. name_regex should
                           be a regex string that would match the 'runtime' name of the metric.
        :type name_regex: string
        :param description: description of the metric (to be used for docs)
        :type description: string
        :param unit: unit in which the progress is reported. Defaults to '%'
        :type unit: string
        """

        super(ProgressMetricDefinition, self).__init__(name=name, m_type=PROGRESS_METRIC_TYPE,
                                                       emission_frequencies=emission_frequencies,
                                                       default_frequency=default_frequency,
                                                       tunable=False, name_regex=name_regex, description=description)
        self.unit = Preconditions.check_not_none(unit)

    def __eq__(self, other):
        ret_val = super(ProgressMetricDefinition, self).__eq__(other=other)
        return ret_val and self.unit == other.unit
