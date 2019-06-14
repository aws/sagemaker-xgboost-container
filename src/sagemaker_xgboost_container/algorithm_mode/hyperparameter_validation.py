from sagemaker_algorithm_toolkit import exceptions as exc
from sagemaker_algorithm_toolkit import hyperparameter_validation as hpv


def initialize(metrics):
    @hpv.range_validator(["auto", "exact", "approx", "hist"])
    def tree_method_range_validator(CATEGORIES, value):
        if "gpu" in value:
            raise exc.UserError("GPU training is not supported yet.")
        return value in CATEGORIES

    @hpv.dependencies_validator(["booster", "process_type"])
    def updater_validator(value, dependencies):
        valid_tree_plugins = ['grow_colmaker', 'distcol', 'grow_histmaker', 'grow_local_histmaker',
                              'grow_skmaker', 'sync', 'refresh', 'prune']
        valid_tree_build_plugins = ['grow_colmaker', 'distcol', 'grow_histmaker',
                                    'grow_local_histmaker', 'grow_colmaker']
        valid_linear_plugins = ['shotgun', 'coord_descent']
        valid_process_update_plugins = ['refresh', 'prune']

        if dependencies.get('booster') == 'gblinear':
            # validate only one linear updater is selected
            if not (len(value) == 1 and value[0] in valid_linear_plugins):
                raise exc.UserError("Linear updater should be one of these options: 'shotgun', 'coor_descent'.")
        elif dependencies.get('process_type') == 'update':
            if not all(x in valid_process_update_plugins for x in value):
                raise exc.UserError("process_type 'update' can only be used with updater 'refresh' and 'prune'")
        else:
            if not all(x in valid_tree_plugins for x in value):
                raise exc.UserError(
                    "Tree updater should be selected from these options: 'grow_colmaker', 'distcol', 'grow_histmaker', "
                    "'grow_local_histmaker', 'grow_skmaker', 'sync', 'refresh', 'prune', 'shortgun', 'coord_descent'.")
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

    @hpv.range_validator(["rmse", "mae", "logloss", "error", "merror", "mlogloss", "auc", "ndcg", "map",
                          "poisson-nloglik", "gamma-nloglik", "gamma-deviance", "tweedie-nloglik"])
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
            if (dependencies["objective"] not in
                ["binary:logistic", "binary:logitraw", "multi:softmax", "multi:softprob",
                 "reg:logistic", "rank:pairwise", "binary:hinge"]):
                raise exc.UserError("Metric 'auc' can only be applied for classification and ranking problem.")

    hyperparameters = hpv.Hyperparameters(
        hpv.IntegerHyperparameter(name="num_round", required=True,
                                  range=hpv.Interval(min_closed=1),
                                  tunable=True, tunable_recommended_range=hpv.Interval(
                                      min_closed=1,
                                      max_closed=4000,
                                      scale=hpv.Interval.LINEAR_SCALE)),
        hpv.IntegerHyperparameter(name="csv_weights", range=hpv.Interval(min_closed=0, max_closed=1), default=0),
        hpv.IntegerHyperparameter(name="early_stopping_rounds", range=hpv.Interval(min_closed=1), required=False),
        hpv.CategoricalHyperparameter(name="booster", range=["gbtree", "gblinear", "dart"], default="gbtree"),
        hpv.IntegerHyperparameter(name="silent", range=hpv.Interval(min_closed=0, max_closed=1), default=1),
        hpv.IntegerHyperparameter(name="nthread", range=hpv.Interval(min_closed=1), required=False),
        hpv.ContinuousHyperparameter(name="eta", range=hpv.Interval(min_closed=0, max_closed=1), default=0.3,
                                     tunable=True,
                                     tunable_recommended_range=hpv.Interval(min_closed=0.1, max_closed=0.5,
                                                                            scale=hpv.Interval.LINEAR_SCALE)),
        hpv.ContinuousHyperparameter(name="gamma", range=hpv.Interval(min_closed=0), default=0,
                                     tunable=True, tunable_recommended_range=hpv.Interval(
                                         min_closed=0, max_closed=5,
                                         scale=hpv.Interval.LINEAR_SCALE)),
        hpv.IntegerHyperparameter(name="max_depth", range=hpv.Interval(min_closed=0), default=6,
                                  tunable=True, tunable_recommended_range=hpv.Interval(
                                      min_closed=0, max_closed=10,
                                      scale=hpv.Interval.LINEAR_SCALE)),
        hpv.ContinuousHyperparameter(name="min_child_weight", range=hpv.Interval(min_closed=0), default=1,
                                     tunable=True,
                                     tunable_recommended_range=hpv.Interval(min_closed=0, max_closed=120,
                                                                            scale=hpv.Interval.LINEAR_SCALE)),
        hpv.ContinuousHyperparameter(name="max_delta_step", range=hpv.Interval(min_closed=0), default=0,
                                     tunable=True, tunable_recommended_range=hpv.Interval(
                                         min_closed=0, max_closed=10,
                                         scale=hpv.Interval.LINEAR_SCALE)),
        hpv.ContinuousHyperparameter(name="subsample", range=hpv.Interval(min_open=0, max_closed=1), default=1,
                                     tunable=True,
                                     tunable_recommended_range=hpv.Interval(min_closed=0.5, max_closed=1,
                                                                            scale=hpv.Interval.LINEAR_SCALE)),
        hpv.ContinuousHyperparameter(name="colsample_bytree", range=hpv.Interval(min_open=0, max_closed=1), default=1,
                                     tunable=True,
                                     tunable_recommended_range=hpv.Interval(min_closed=0.5, max_closed=1,
                                                                            scale=hpv.Interval.LINEAR_SCALE)),
        hpv.ContinuousHyperparameter(name="colsample_bylevel", range=hpv.Interval(min_open=0, max_closed=1), default=1,
                                     tunable=True,
                                     tunable_recommended_range=hpv.Interval(min_closed=0.1, max_closed=1,
                                                                            scale=hpv.Interval.LINEAR_SCALE)),
        hpv.ContinuousHyperparameter(name="lambda", range=hpv.Interval(min_closed=0), default=1,
                                     tunable=True,
                                     tunable_recommended_range=hpv.Interval(min_closed=0, max_closed=1000,
                                                                            scale=hpv.Interval.LINEAR_SCALE)),
        hpv.ContinuousHyperparameter(name="alpha", default=0, range=hpv.Interval(min_closed=0),
                                     tunable=True,
                                     tunable_recommended_range=hpv.Interval(min_closed=0, max_closed=1000,
                                                                            scale=hpv.Interval.LINEAR_SCALE)),
        hpv.CategoricalHyperparameter(name="tree_method", range=tree_method_range_validator, default="auto"),
        hpv.ContinuousHyperparameter(name="sketch_eps", range=hpv.Interval(min_open=0, max_open=1), default=0.03),
        hpv.ContinuousHyperparameter(name="scale_pos_weight", range=hpv.Interval(min_open=0), default=1),
        hpv.CommaSeparatedListHyperparameter(name="updater",
                                             range=['grow_colmaker', 'distcol',
                                                    'grow_histmaker', 'grow_local_histmaker',
                                                    'grow_skmaker', 'sync', 'refresh', 'prune',
                                                    'grow_colmaker', 'distcol', 'grow_histmaker',
                                                    'grow_local_histmaker', 'grow_colmaker',
                                                    'shotgun', 'coord_descent',
                                                    'refresh', 'prune'],
                                             dependencies=updater_validator,
                                             default="grow_colmaker,prune"),
        hpv.CategoricalHyperparameter(name="dsplit", range=["row", "col"], default="row"),
        hpv.IntegerHyperparameter(name="refresh_leaf", range=hpv.Interval(min_closed=0, max_closed=1), default=1),
        hpv.CategoricalHyperparameter(name="process_type", range=["default", "update"],
                                      default="default"),
        hpv.CategoricalHyperparameter(name="grow_policy", range=["depthwise", "lossguide"], default="depthwise"),
        hpv.IntegerHyperparameter(name="max_leaves", range=hpv.Interval(min_closed=0), default=0),
        hpv.IntegerHyperparameter(name="max_bin", range=hpv.Interval(min_closed=0), default=256),
        hpv.CategoricalHyperparameter(name="predictor", range=predictor_validator, default="cpu_predictor"),
        hpv.CategoricalHyperparameter(name="sample_type", range=["uniform", "weighted"], default="uniform"),
        hpv.CategoricalHyperparameter(name="normalize_type", range=["tree", "forest"], default="tree"),
        hpv.ContinuousHyperparameter(name="rate_drop", range=hpv.Interval(min_closed=0, max_closed=1), default=0.0),
        hpv.IntegerHyperparameter(name="one_drop", range=hpv.Interval(min_closed=0, max_closed=1), default=0),
        hpv.ContinuousHyperparameter(name="skip_drop", range=hpv.Interval(min_closed=0, max_closed=1), default=0.0),
        hpv.ContinuousHyperparameter(name="lambda_bias", range=hpv.Interval(min_closed=0, max_closed=1), default=0.0),
        hpv.ContinuousHyperparameter(name="tweedie_variance_power", range=hpv.Interval(min_open=1, max_open=2),
                                     default=1.5),
        hpv.CategoricalHyperparameter(name="objective",
                                      range=["reg:linear", "reg:logistic", "binary:logistic", "binary:logitraw",
                                             "count:poisson", "multi:softmax", "multi:softprob", "rank:pairwise",
                                             "reg:gamma", "reg:tweedie"],
                                      dependencies=objective_validator,
                                      default="reg:linear"),
        hpv.IntegerHyperparameter(name="num_class",
                                  range=hpv.Interval(min_closed=2),
                                  required=False),
        hpv.ContinuousHyperparameter(name="base_score", range=hpv.Interval(min_closed=0), default=0.5),
        hpv.CategoricalHyperparameter(name="_tuning_objective_metric", range=metrics.names, required=False),
        hpv.CommaSeparatedListHyperparameter(name="eval_metric",
                                             range=eval_metric_range_validator,
                                             dependencies=eval_metric_dep_validator,
                                             required=False),
        hpv.IntegerHyperparameter(name="seed", range=hpv.Interval(min_open=-2**31, max_open=2**31-1),
                                  default=0),
        )
    return hyperparameters
