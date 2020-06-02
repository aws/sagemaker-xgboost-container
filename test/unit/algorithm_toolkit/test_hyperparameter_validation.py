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
import unittest

from sagemaker_algorithm_toolkit import exceptions as exc
from sagemaker_algorithm_toolkit import hyperparameter_validation as hpv


@hpv.dependencies_validator(["index_type"])
def faiss_index_pq_m_validator(value, index_type):
    if index_type != "faiss.IVFPQ" and value is not None:
        raise ValueError("Hyperparameter 'faiss_index_pq_m' should only be set when 'index_type' = 'faiss.IVFPQ'.")


@hpv.dependencies_validator(["dimension_reduction_type", "feature_dim"])
def dimension_reduction_target_validator(value, dependencies):
    dimension_reduction_type, feature_dim = dependencies["dimension_reduction_type"], dependencies["feature_dim"]
    if dimension_reduction_type == "none" and value is not None:
        raise ValueError("Hyperparameter 'dimension_reduction_target' should only be set when "
                         "'dimension_reduction_type' is set.")
    if value is not None and value >= feature_dim:
        raise ValueError("Hyperparameter 'dimension_reduction_target' should be less than 'feature_dim'.")


@hpv.range_validator([hpv.Interval(min_open=0), "auto"])
def faiss_index_ivf_nlists_validator(RANGE, value):
    return value == RANGE[1] or int(value) in RANGE[0]


def initialize():
    return hpv.Hyperparameters(
        hpv.IntegerHyperparameter(name="feature_dim", range=hpv.Interval(min_open=0), required=True),
        hpv.IntegerHyperparameter(name="mini_batch_size", range=hpv.Interval(min_open=0), default=5000),
        hpv.IntegerHyperparameter(name="k", range=hpv.Interval(min_open=0, max_closed=1024), required=True,
                                  tunable=True),
        hpv.CategoricalHyperparameter(name="predictor_type", range=["classifier", "regressor"], required=True),
        # dimension reduction
        hpv.CategoricalHyperparameter(name="dimension_reduction_type", range=["none", "sign", "fjlt"], default="none"),
        hpv.IntegerHyperparameter(name="dimension_reduction_target",
                                  range=hpv.Interval(min_open=0),
                                  dependencies=dimension_reduction_target_validator,
                                  required=False),
        # sampling
        hpv.IntegerHyperparameter(name="sample_size", range=hpv.Interval(min_open=0), required=True, tunable=True),
        # index
        hpv.CategoricalHyperparameter(name="index_type",
                                      range=["faiss.Flat", "faiss.IVFFlat", "faiss.IVFPQ"], default="faiss.Flat"),
        hpv.CategoricalHyperparameter(name="index_metric", range=["L2", "INNER_PRODUCT", "COSINE"], default="L2"),
        hpv.IntegerHyperparameter(name="faiss_index_pq_m",
                                  range=[1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 96],
                                  dependencies=faiss_index_pq_m_validator,
                                  required=False),
        hpv.Hyperparameter(name="faiss_index_ivf_nlists", range=faiss_index_ivf_nlists_validator, default="auto"),
        # tuning
        hpv.Hyperparameter(name="_tuning_objective_metric", required=False))


class TestHyperparameters(unittest.TestCase):
    def test_simple_dependency(self):
        @hpv.dependencies_validator(["b"])
        def validate(value, dependencies):
            pass
        hps = hpv.Hyperparameters(hpv.Hyperparameter(name="a", dependencies=validate, required=True),
                                  hpv.Hyperparameter(name="b", dependencies=None, required=True))
        result = hps.validate({"a": "5", "b": "lol"})
        self.assertEqual(result["a"], "5")
        self.assertEqual(result["b"], "lol")

    def test_default(self):
        hps = hpv.Hyperparameters(hpv.Hyperparameter(name="a", default=True))

        result = hps.validate({})
        self.assertTrue(result['a'])

    def test_simple_integral(self):
        hps = hpv.Hyperparameters(hpv.IntegerHyperparameter(name="a", range=hpv.Interval(min_open=0, max_closed=1),
                                                            required=False))
        with self.assertRaises(exc.UserError):
            hps.validate({"a": "5"})

        result = hps.validate({"a": "1"})
        self.assertEqual(result["a"], 1)

    def test_simple_categorical(self):
        hps = hpv.Hyperparameters(hpv.CategoricalHyperparameter(name="a", range=["classifier", "regressor"],
                                                                required=True))
        with self.assertRaises(exc.UserError):
            hps.validate({})

        with self.assertRaises(exc.UserError):
            hps.validate({"a": "nope"})

        result = hps.validate({"a": "classifier"})
        self.assertEqual(result["a"], "classifier")

    def test_simple_continuous(self):
        hps = hpv.Hyperparameters(hpv.ContinuousHyperparameter(name="gamma",
                                                               range=hpv.Interval(min_closed=0, max_closed=1),
                                                               required=True))
        with self.assertRaises(exc.UserError):
            hps.validate({})

        with self.assertRaises(exc.UserError):
            hps.validate({"gamma": "ha"})

        with self.assertRaises(exc.UserError):
            hps.validate({"gamma": "-1"})

        result = hps.validate({"gamma": "0.667"})
        self.assertEqual(result["gamma"], 0.667)

    def test_complex_knn(self):
        hps = initialize()
        result = hps.validate({"feature_dim": "4",
                               "k": "100",
                               "predictor_type": "regressor",
                               "sample_size": "256"})
        self.assertEqual(result["feature_dim"], 4)
        self.assertEqual(result["k"], 100)
        self.assertEqual(result["predictor_type"], "regressor")
        self.assertEqual(result["sample_size"], 256)

    def test_integer_tunable_range(self):
        hyperparameter = hpv.IntegerHyperparameter(
            name="x",
            range=hpv.Interval(min_open=0),
            tunable=True,
            tunable_recommended_range=hpv.Interval(
                min_open=0, max_open=100, scale=hpv.Interval.LINEAR_SCALE),
            required=True)
        self.assertEqual(hyperparameter.format_tunable_range(),
                         {'IntegerParameterRanges': [{'MaxValue': '100',
                                                      'MinValue': '0',
                                                      'Name': 'x',
                                                      'ScalingType': 'Linear'}]})

    def test_categorical_tunable_range(self):
        hyperparameter = hpv.CategoricalHyperparameter(
            name="x", range=["a", "b", "c"],
            tunable=True, tunable_recommended_range=["a", "b"],
            required=True)
        self.assertEqual(hyperparameter.format_tunable_range(),
                         {'CategoricalParameterRanges': [
                             {'Name': 'x', 'Values': ['a', 'b']}]})

    def test_continuous_tunable_range(self):
        hyperparameter = hpv.ContinuousHyperparameter(
            name="x", range=hpv.Interval(min_open=0),
            tunable=True,
            tunable_recommended_range=hpv.Interval(
                min_open=0, max_open=100, scale=hpv.Interval.LINEAR_SCALE),
            required=True)
        self.assertEqual(hyperparameter.format_tunable_range(),
                         {'ContinuousParameterRanges': [{'MaxValue': '100',
                                                         'MinValue': '0',
                                                         'Name': 'x',
                                                         'ScalingType': 'Linear'}]})
