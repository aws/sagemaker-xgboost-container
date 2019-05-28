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
from collections import defaultdict
from collections import namedtuple

from sagemaker_xgboost_container.metrics.utils import Preconditions
from sagemaker_xgboost_container.exceptions import AlgorithmError, CustomerValueError

from sagemaker_xgboost_container.metrics.performance_metric_definitions import QualityMetricDefinition, \
    SCORE_METRIC_TYPE, FREQUENCY_PER_EPOCH, LOSS_METRIC_TYPE, PROGRESS_METRIC_TYPE, THROUGHPUT_METRIC_TYPE

from sagemaker_xgboost_container.constants.xgb_constants import TRAIN_CHANNEL, VAL_CHANNEL, _SEPARATOR

MetricNameComponents = namedtuple('MetricNameComponents',
                                  ['data_segment', 'metric_name', 'emission_frequency'])
MetricNameComponents.__new__.__defaults__ = (None,) * len(MetricNameComponents._fields)


class XGBoostPerformanceMetrics(object):
    """
    Register and validate HPO metrics for XGBoost
    """

    def __init__(self, hyperparameters=None, data_config=None):
        """
        Initializes the metrics. Calls the method register_all_allowed_metrics and
        runs validation checks. It sets the objective to be runtime if the hyperparameters is not None.
        :param hyperparameters: training job hyperparameters.
        :param data_config: dict for data configuration.
        """
        self._ALLOWED_PROGRESS_METRICS = defaultdict(dict)
        self._ALLOWED_THROUGHPUT_METRICS = defaultdict(dict)
        self._ALLOWED_QUALITY_METRICS = defaultdict(dict)
        self.is_runtime = hyperparameters is not None and data_config is not None

        self._data_segments = {}
        self.tuning_objective_metric = None

        if self.is_runtime:
            self._data_segments = self.get_channel_names(data_config)
            self.tuning_objective_metric = hyperparameters.get('_tuning_objective_metric')

        self.register_all_allowed_metrics(hyperparameters)
        self.validate_objective_metric()

    def get_channel_names(self, data_config):
        """
        return available channel names
        """
        return data_config.keys()

    def register_all_allowed_metrics(self, hyperparameters=None):
        """
        Only support the most common metrics at the moment due to limitation on HPO
        TODO: Add all the supported metrics on XGBoost
        """
        metric_rmse = QualityMetricDefinition(name="rmse", m_type=LOSS_METRIC_TYPE,
                                              emission_frequencies={FREQUENCY_PER_EPOCH}, tunable=True,
                                              description="Root Mean Square Error")

        metric_mae = QualityMetricDefinition(name="mae", m_type=LOSS_METRIC_TYPE,
                                             emission_frequencies={FREQUENCY_PER_EPOCH}, tunable=True,
                                             description="Mean Absolute Error")

        metric_logloss = QualityMetricDefinition(name="logloss", m_type=LOSS_METRIC_TYPE,
                                                 emission_frequencies={FREQUENCY_PER_EPOCH}, tunable=True,
                                                 description="Negative Log-likelihood")

        metric_error = QualityMetricDefinition(name="error", m_type=LOSS_METRIC_TYPE,
                                               emission_frequencies={FREQUENCY_PER_EPOCH}, tunable=True,
                                               description="Binary Classification Error Rate")

        metric_merror = QualityMetricDefinition(name="merror", m_type=LOSS_METRIC_TYPE,
                                                emission_frequencies={FREQUENCY_PER_EPOCH}, tunable=True,
                                                description="Multiclass Classification Error Rate")

        metric_mlogloss = QualityMetricDefinition(name="mlogloss", m_type=LOSS_METRIC_TYPE,
                                                  emission_frequencies={FREQUENCY_PER_EPOCH}, tunable=True,
                                                  description="Multiclass Logloss")

        metric_auc = QualityMetricDefinition(name="auc", m_type=SCORE_METRIC_TYPE,
                                             emission_frequencies={FREQUENCY_PER_EPOCH}, tunable=True,
                                             description="Area Under the Curve")

        metric_ndcg = QualityMetricDefinition(name="ndcg", m_type=SCORE_METRIC_TYPE,
                                              emission_frequencies={FREQUENCY_PER_EPOCH}, tunable=True,
                                              description="Normalized Discounted Cumulative Gain")

        metric_map = QualityMetricDefinition(name="map", m_type=SCORE_METRIC_TYPE,
                                             emission_frequencies={FREQUENCY_PER_EPOCH}, tunable=True,
                                             description="Mean average precision")

        self.register_allowed_metric(TRAIN_CHANNEL, metric_rmse)
        self.register_allowed_metric(TRAIN_CHANNEL, metric_mae)
        self.register_allowed_metric(TRAIN_CHANNEL, metric_logloss)
        self.register_allowed_metric(TRAIN_CHANNEL, metric_error)
        self.register_allowed_metric(TRAIN_CHANNEL, metric_merror)
        self.register_allowed_metric(TRAIN_CHANNEL, metric_mlogloss)
        self.register_allowed_metric(TRAIN_CHANNEL, metric_auc)
        self.register_allowed_metric(TRAIN_CHANNEL, metric_ndcg)
        self.register_allowed_metric(TRAIN_CHANNEL, metric_map)
        self.register_allowed_metric(VAL_CHANNEL, metric_rmse)
        self.register_allowed_metric(VAL_CHANNEL, metric_mae)
        self.register_allowed_metric(VAL_CHANNEL, metric_logloss)
        self.register_allowed_metric(VAL_CHANNEL, metric_error)
        self.register_allowed_metric(VAL_CHANNEL, metric_merror)
        self.register_allowed_metric(VAL_CHANNEL, metric_mlogloss)
        self.register_allowed_metric(VAL_CHANNEL, metric_auc)
        self.register_allowed_metric(VAL_CHANNEL, metric_ndcg)
        self.register_allowed_metric(VAL_CHANNEL, metric_map)

    def register_allowed_metric(self, data_segment, metric_def):
        """
        This method  can be used to update all allowed metrics.
        :param data_segment: data_segment to report the metric on
        :type data_segment: string
        :param metric_def: PerformanceMetricDefinition
        :type metric_def: PerformanceMetricDefinition
        """
        Preconditions.check_not_none(metric_def)
        Preconditions.check_not_none(data_segment)

        # do not add the metric if the requested data segment is not provided
        if self.is_runtime and data_segment not in self._data_segments:
            return

        map_to_fill = self._ALLOWED_QUALITY_METRICS
        if metric_def.type == PROGRESS_METRIC_TYPE:
            map_to_fill = self._ALLOWED_PROGRESS_METRICS
        elif metric_def.type == THROUGHPUT_METRIC_TYPE:
            map_to_fill = self._ALLOWED_THROUGHPUT_METRICS

        map_to_fill[data_segment].update({metric_def.name: metric_def})

        # cache by runtime name as well to facilitate querying metric type.
        # since runtime name may vary from the metric name if the metric is parameterized.
        if self.is_runtime and metric_def.name_regex and metric_def.name != metric_def.name_regex:
            map_to_fill[data_segment].update({metric_def.name_regex: metric_def})

    def validate_objective_metric(self):
        """
        Checks if the metric name passed via hyperparameter '_tuning_objective_metric' is indeed allowed to be an
        objective metric. It raises CustomerValueError if the metric is not supported or not an objective metric.
        :raises: CustomerValueError if the metric is not an allowed objective metric.
        """
        if not self.tuning_objective_metric:
            return

        mnc = self.decode_metric_name(self.tuning_objective_metric)
        if not mnc:
            raise CustomerValueError('The metric name {} could not be decoded. Please '
                                     'check the name and retry.'.format(self.tuning_objective_metric))

        ds = self._ALLOWED_QUALITY_METRICS.get(mnc.data_segment)
        if not ds:
            raise CustomerValueError('Objective metric {} is reported on the {ds} data segment. '
                                     'Data channel {ds} is not provided. Please consider adding it to '
                                     'InputDataConfig and try again.'.format(self.tuning_objective_metric,
                                                                             ds=mnc.data_segment))
        m = ds.get(mnc.metric_name)
        if not m:
            raise CustomerValueError('The metric {} is not a supported metric. Please set a valid '
                                     'objective metric name and retry.'.format(self.tuning_objective_metric))
        if not m.tunable:
            raise CustomerValueError('The metric {} can not be used as objective metric. Please set a '
                                     'valid objective metric name and retry.'.format(self.tuning_objective_metric))

    @classmethod
    def decode_metric_name(cls, encoded_metric_name):
        """Decode a metric name to MetricNameComponents.

        Raise Value error if encoded_metric_name is None, empty string or is malformed.

        :param encoded_metric_name: metric name encoded via encode_metric_name
        :type encoded_metric_name: string
        :return: MetricNameComponents containing decoded components
        """
        if encoded_metric_name:
            tokens = encoded_metric_name.split(_SEPARATOR)
        else:
            tokens = []
        num_comps = len(tokens)
        if num_comps < 2 or num_comps > 3:
            raise CustomerValueError('Expected encoded metric string to '
                                     'have 2 or 3 components separated '
                                     'by {}. Got {}.'.format(_SEPARATOR, encoded_metric_name))
        return MetricNameComponents(*tokens)


_SCHEMA = None
_RUNTIME = None


def get_schema():
    global _SCHEMA
    if not _SCHEMA:
        _SCHEMA = XGBoostPerformanceMetrics()
    return _SCHEMA


def create_runtime(hyperparameters, data_config):
    global _RUNTIME
    if not _RUNTIME:
        _RUNTIME = XGBoostPerformanceMetrics(hyperparameters, data_config)


def get_runtime():
    """

    :rtype: object
    """
    if not _RUNTIME:
        raise AlgorithmError('Performance metric runtime cannot be instantiated without hyperparameters.')

    return _RUNTIME
