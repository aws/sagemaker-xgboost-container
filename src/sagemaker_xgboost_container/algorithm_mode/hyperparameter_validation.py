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
from sagemaker_algorithm_toolkit import hyperparameter_validation as hpv

from sagemaker_xgboost_container.constants.xgb_constants import XGB_MAXIMIZE_METRICS, XGB_MINIMIZE_METRICS


def initialize(metrics):
    @hpv.range_validator(["auto", "exact", "approx", "hist"])
    def tree_method_range_validator(CATEGORIES, value):
        if "gpu" in value:
            raise exc.UserError("GPU training is not supported yet.")
        return value in CATEGORIES

    @hpv.dependencies_validator(["booster", "process_type"])
    def updater_validator(value, dependencies):
        valid_tree_plugins = ['grow_colmaker', 'distcol', 'grow_histmaker', 'grow_local_histmaker',
                              'grow_skmaker', 'sync', 'refresh', 'prune', 'grow_quantile_histmaker']
        valid_tree_build_plugins = ['grow_colmaker', 'distcol', 'grow_histmaker',
                                    'grow_local_histmaker', 'grow_colmaker', 'grow_quantile_histmaker']
        valid_linear_plugins = ['shotgun', 'coord_descent']
        valid_process_update_plugins = ['refresh', 'prune']

        if dependencies.get('booster') == 'gblinear':
            # validate only one linear updater is selected
            if not (len(value) == 1 and value[0] in valid_linear_plugins):
                raise exc.UserError("Linear updater should be one of these options: {}.".format(
                    ', '.join("'{0}'".format(valid_updater for valid_updater in valid_linear_plugins))
                ))
        elif dependencies.get('process_type') == 'update':
            if not all(x in valid_process_update_plugins for x in value):
                raise exc.UserError("process_type 'update' can only be used with updater 'refresh' and 'prune'")
        else:
            if not all(x in valid_tree_plugins for x in value):
                raise exc.UserError(
                    "Tree updater should be selected from these options: 'grow_colmaker', 'distcol', 'grow_histmaker', "
                    "'grow_local_histmaker', 'grow_skmaker', 'grow_quantile_histmaker', 'sync', 'refresh', 'prune', "
                    "'shortgun', 'coord_descent'.")
            # validate only one tree updater is selected
            counter = 0
            for tmp in value:
                if tmp in valid_tree_build_plugins:
                    counter += 1
            if counter > 1:
                raise exc.UserError("Only one tree grow plugin can be selected. Choose one from the"
                                    "following: 'grow_colmaker', 'distcol', 'grow_histmaker', "
                                    "'grow_local_histmaker', 'grow_skmaker'")

    @hpv.range_validator(["cpu_predictor"])
    def predictor_validator(CATEGORIES, value):
        if "gpu" in value:
            raise exc.UserError("GPU training is not supported yet.")
        return value in CATEGORIES

    @hpv.dependencies_validator(["num_class"])
    def objective_validator(value, dependencies):
        num_class = dependencies.get("num_class")
        if value in ("multi:softmax", "multi:softprob") and num_class is None:
            raise exc.UserError("Require input for parameter 'num_class' for multi-classification")
        if value is None and num_class is not None:
            raise exc.UserError("Do not need to setup parameter 'num_class' for learning task other than "
                                "multi-classification.")

    @hpv.range_validator(XGB_MAXIMIZE_METRICS + XGB_MINIMIZE_METRICS)
    def eval_metric_range_validator(SUPPORTED_METRIC, metric):
        if "<function" in metric:
            raise exc.UserError("User defined evaluation metric {} is not supported yet.".format(metric))

        if "@" in metric:
            metric_name = metric.split('@')[0].strip()
            metric_threshold = metric.split('@')[1].strip()
            if metric_name not in ["error", "ndcg", "map"]:
                raise exc.UserError(
                    "Metric '{}' is not supported. Parameter 'eval_metric' with customized threshold should "
                    "be one of these options: 'error', 'ndcg', 'map'.".format(metric))
            try:
                float(metric_threshold)
            except ValueError:
                raise exc.UserError("Threshold value 't' in '{}@t' expects float input.".format(metric_name))
            return True

        return metric in SUPPORTED_METRIC

    @hpv.dependencies_validator(["objective"])
    def eval_metric_dep_validator(value, dependencies):
        if "auc" in value:
            if not any(dependencies["objective"].startswith(metric_type) for metric_type in [
                    'binary:', 'rank:']):
                raise exc.UserError("Metric 'auc' can only be applied for classification and ranking problems.")

    @hpv.dependencies_validator(["tree_method"])
    def monotone_constraints_validator(value, dependencies):
        tree_method = dependencies.get("tree_method")
        if value is not None and tree_method not in ("exact", "hist"):
            raise exc.UserError("monotone_constraints can be used only when the tree_method parameter is set to "
                                "either 'exact' or 'hist'.")

    @hpv.dependencies_validator(["tree_method"])
    def interaction_constraints_validator(value, dependencies):
        tree_method = dependencies.get("tree_method")
        if value is not None and tree_method not in ("exact", "hist", "approx"):
            raise exc.UserError("interaction_constraints can be used only when the tree_method parameter is set to "
                                "either 'exact', 'hist' or 'approx'.")

    hyperparameters = hpv.Hyperparameters(
        hpv.IntegerHyperparameter(name="num_round", required=True,
                                  range=hpv.Interval(min_closed=1),
                                  tunable=True, tunable_recommended_range=hpv.Interval(
                                      min_closed=1,
                                      max_closed=4000,
                                      scale=hpv.Interval.LINEAR_SCALE)),
        hpv.IntegerHyperparameter(name="csv_weights", range=hpv.Interval(min_closed=0, max_closed=1), required=False),
        hpv.IntegerHyperparameter(name="early_stopping_rounds", range=hpv.Interval(min_closed=1), required=False),
        hpv.CategoricalHyperparameter(name="booster", range=["gbtree", "gblinear", "dart"], required=False),
        hpv.IntegerHyperparameter(name="silent", range=hpv.Interval(min_closed=0, max_closed=1), required=False),
        hpv.IntegerHyperparameter(name="verbosity", range=hpv.Interval(min_closed=0, max_closed=3), required=False),
        hpv.IntegerHyperparameter(name="nthread", range=hpv.Interval(min_closed=1), required=False),
        hpv.ContinuousHyperparameter(name="eta", range=hpv.Interval(min_closed=0, max_closed=1), required=False,
                                     tunable=True,
                                     tunable_recommended_range=hpv.Interval(min_closed=0.1, max_closed=0.5,
                                                                            scale=hpv.Interval.LINEAR_SCALE)),
        hpv.ContinuousHyperparameter(name="gamma", range=hpv.Interval(min_closed=0), required=False,
                                     tunable=True, tunable_recommended_range=hpv.Interval(
                                         min_closed=0, max_closed=5,
                                         scale=hpv.Interval.LINEAR_SCALE)),
        hpv.IntegerHyperparameter(name="max_depth", range=hpv.Interval(min_closed=0), required=False,
                                  tunable=True, tunable_recommended_range=hpv.Interval(
                                      min_closed=0, max_closed=10,
                                      scale=hpv.Interval.LINEAR_SCALE)),
        hpv.ContinuousHyperparameter(name="min_child_weight", range=hpv.Interval(min_closed=0), required=False,
                                     tunable=True,
                                     tunable_recommended_range=hpv.Interval(min_closed=0, max_closed=120,
                                                                            scale=hpv.Interval.LINEAR_SCALE)),
        hpv.ContinuousHyperparameter(name="max_delta_step", range=hpv.Interval(min_closed=0), required=False,
                                     tunable=True, tunable_recommended_range=hpv.Interval(
                                         min_closed=0, max_closed=10,
                                         scale=hpv.Interval.LINEAR_SCALE)),
        hpv.ContinuousHyperparameter(name="subsample", range=hpv.Interval(min_open=0, max_closed=1), required=False,
                                     tunable=True,
                                     tunable_recommended_range=hpv.Interval(min_closed=0.5, max_closed=1,
                                                                            scale=hpv.Interval.LINEAR_SCALE)),
        hpv.ContinuousHyperparameter(name="colsample_bytree", range=hpv.Interval(min_open=0, max_closed=1),
                                     required=False, tunable=True,
                                     tunable_recommended_range=hpv.Interval(min_closed=0.5, max_closed=1,
                                                                            scale=hpv.Interval.LINEAR_SCALE)),
        hpv.ContinuousHyperparameter(name="colsample_bylevel", range=hpv.Interval(min_open=0, max_closed=1),
                                     required=False, tunable=True,
                                     tunable_recommended_range=hpv.Interval(min_closed=0.1, max_closed=1,
                                                                            scale=hpv.Interval.LINEAR_SCALE)),
        hpv.ContinuousHyperparameter(name="colsample_bynode", range=hpv.Interval(min_open=0, max_closed=1),
                                     required=False, tunable=True,
                                     tunable_recommended_range=hpv.Interval(min_closed=0.1, max_closed=1,
                                                                            scale=hpv.Interval.LINEAR_SCALE)),
        hpv.ContinuousHyperparameter(name="lambda", range=hpv.Interval(min_closed=0), required=False,
                                     tunable=True,
                                     tunable_recommended_range=hpv.Interval(min_closed=0, max_closed=1000,
                                                                            scale=hpv.Interval.LINEAR_SCALE)),
        hpv.ContinuousHyperparameter(name="alpha", range=hpv.Interval(min_closed=0), required=False,
                                     tunable=True,
                                     tunable_recommended_range=hpv.Interval(min_closed=0, max_closed=1000,
                                                                            scale=hpv.Interval.LINEAR_SCALE)),
        hpv.CategoricalHyperparameter(name="tree_method", range=tree_method_range_validator, required=False),
        hpv.ContinuousHyperparameter(name="sketch_eps", range=hpv.Interval(min_open=0, max_open=1), required=False),
        hpv.ContinuousHyperparameter(name="scale_pos_weight", range=hpv.Interval(min_open=0), required=False),
        hpv.CommaSeparatedListHyperparameter(name="updater",
                                             range=['grow_colmaker', 'distcol',
                                                    'grow_histmaker', 'grow_local_histmaker',
                                                    'grow_skmaker', 'sync', 'refresh', 'prune',
                                                    'grow_colmaker', 'distcol', 'grow_histmaker',
                                                    'grow_local_histmaker', 'grow_colmaker',
                                                    'shotgun', 'coord_descent',
                                                    'refresh', 'prune'],
                                             dependencies=updater_validator,
                                             required=False),
        hpv.CategoricalHyperparameter(name="dsplit", range=["row", "col"], required=False),
        hpv.IntegerHyperparameter(name="refresh_leaf", range=hpv.Interval(min_closed=0, max_closed=1), required=False),
        hpv.CategoricalHyperparameter(name="process_type", range=["default", "update"], required=False),
        hpv.CategoricalHyperparameter(name="grow_policy", range=["depthwise", "lossguide"], required=False),
        hpv.IntegerHyperparameter(name="max_leaves", range=hpv.Interval(min_closed=0), required=False),
        hpv.IntegerHyperparameter(name="max_bin", range=hpv.Interval(min_closed=0), required=False),
        hpv.CategoricalHyperparameter(name="predictor", range=predictor_validator, required=False),
        hpv.TupleHyperparameter(name="monotone_constraints", range=[-1, 0, 1], required=False,
                                dependencies=monotone_constraints_validator),
        hpv.NestedListHyperparameter(name="interaction_constraints", range=hpv.Interval(min_closed=1), required=False,
                                     dependencies=interaction_constraints_validator),
        hpv.CategoricalHyperparameter(name="sample_type", range=["uniform", "weighted"], required=False),
        hpv.CategoricalHyperparameter(name="normalize_type", range=["tree", "forest"], required=False),
        hpv.ContinuousHyperparameter(name="rate_drop", range=hpv.Interval(min_closed=0, max_closed=1), required=False),
        hpv.IntegerHyperparameter(name="one_drop", range=hpv.Interval(min_closed=0, max_closed=1), required=False),
        hpv.ContinuousHyperparameter(name="skip_drop", range=hpv.Interval(min_closed=0, max_closed=1), required=False),
        hpv.ContinuousHyperparameter(name="lambda_bias", range=hpv.Interval(min_closed=0, max_closed=1),
                                     required=False),
        hpv.ContinuousHyperparameter(name="tweedie_variance_power", range=hpv.Interval(min_open=1, max_open=2),
                                     required=False),
        hpv.CategoricalHyperparameter(name="objective",
                                      range=["binary:logistic", "binary:logitraw", "binary:hinge",
                                             "count:poisson", "multi:softmax", "multi:softprob",
                                             "rank:pairwise", "rank:ndcg", "rank:map", "reg:linear",
                                             "reg:squarederror", "reg:logistic", "reg:gamma",
                                             "reg:squaredlogerror", "reg:tweedie", "survival:cox"],
                                      dependencies=objective_validator,
                                      required=False),
        hpv.IntegerHyperparameter(name="num_class",
                                  range=hpv.Interval(min_closed=2),
                                  required=False),
        hpv.ContinuousHyperparameter(name="base_score", range=hpv.Interval(min_closed=0), required=False),
        hpv.CategoricalHyperparameter(name="_tuning_objective_metric", range=metrics.names, required=False),
        hpv.CommaSeparatedListHyperparameter(name="eval_metric",
                                             range=eval_metric_range_validator,
                                             dependencies=eval_metric_dep_validator,
                                             required=False),
        hpv.IntegerHyperparameter(name="seed", range=hpv.Interval(min_open=-2**31, max_open=2**31-1),
                                  required=False),
        )

    hyperparameters.declare_alias("eta", "learning_rate")
    hyperparameters.declare_alias("gamma", "min_split_loss")
    hyperparameters.declare_alias("lambda", "reg_lambda")
    hyperparameters.declare_alias("alpha", "reg_alpha")

    return hyperparameters
