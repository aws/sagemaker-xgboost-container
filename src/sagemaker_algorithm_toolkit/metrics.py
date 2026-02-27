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
import logging

from sagemaker_algorithm_toolkit import exceptions as exc


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
        return {"MetricName": self.name, "Type": self.direction}

    def format_definition(self):
        return {"Name": self.name, "Regex": self.regex}


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


class MetricHistory(object):
    """Record and analyse the value of a single Metric across training iterations."""

    def __init__(self, metric):
        self.metric = metric
        self._values = []
        self._iterations = []

    def record(self, value, iteration=None):
        """Append a new observation. If iteration is None, uses the next integer index."""
        self._values.append(float(value))
        self._iterations.append(
            iteration if iteration is not None else len(self._values) - 1
        )

    @property
    def values(self):
        return list(self._values)

    @property
    def iterations(self):
        return list(self._iterations)

    def best_value(self):
        """Return the best observed value, respecting the metric direction."""
        if not self._values:
            return None
        if self.metric.direction == Metric.MAXIMIZE:
            return max(self._values)
        return min(self._values)

    def best_iteration(self):
        """Return the iteration index at which the best value was observed."""
        if not self._values:
            return None
        if self.metric.direction == Metric.MAXIMIZE:
            idx = self._values.index(max(self._values))
        else:
            idx = self._values.index(min(self._values))
        return self._iterations[idx]

    def rounds_without_improvement(self):
        """Return the number of iterations since the best value was recorded."""
        if len(self._values) < 2:
            return 0
        best_iter = self.best_iteration()
        last_iter = self._iterations[-1]
        return last_iter - best_iter

    def is_improving(self, window=5):
        """Return True if the most recent value is the best within the last `window` steps."""
        if len(self._values) < 2:
            return True
        recent = self._values[-window:]
        if self.metric.direction == Metric.MAXIMIZE:
            return recent[-1] >= max(recent[:-1])
        return recent[-1] <= min(recent[:-1])

    def summary(self):
        """Return a dict with a snapshot of current history statistics."""
        if not self._values:
            return {"name": self.metric.name, "count": 0}
        return {
            "name": self.metric.name,
            "count": len(self._values),
            "last": self._values[-1],
            "best": self.best_value(),
            "best_iteration": self.best_iteration(),
            "rounds_without_improvement": self.rounds_without_improvement(),
        }


class MetricsTracker(object):
    """Track metric values across training iterations for an entire Metrics collection.

    Provides early stopping detection and an XGBoost TrainingCallback factory so
    results from xgb.train() eval sets feed directly into the tracker.

    Example usage with XGBoost::

        tracker = MetricsTracker(metrics, early_stopping_rounds=10)
        bst = xgb.train(
            params, dtrain,
            evals=[(dval, "validation")],
            callbacks=[tracker.xgboost_callback()],
        )
        print(tracker.summary())
    """

    def __init__(self, metrics, early_stopping_rounds=None):
        self.metrics = metrics
        self.early_stopping_rounds = early_stopping_rounds
        self._histories = {
            name: MetricHistory(metrics[name]) for name in metrics.names
        }
        self._current_iteration = 0

    def record(self, metric_values, iteration=None):
        """Record a mapping of {metric_name: value} at the given iteration.

        Unknown metric names are logged as warnings and skipped.
        """
        iter_num = iteration if iteration is not None else self._current_iteration
        for name, value in metric_values.items():
            if name in self._histories:
                self._histories[name].record(value, iter_num)
            else:
                logging.warning(
                    "MetricsTracker received unknown metric '%s', skipping.", name
                )
        self._current_iteration = iter_num + 1

    def should_stop_early(self):
        """Return True if all tunable metrics have stalled for early_stopping_rounds iterations."""
        if self.early_stopping_rounds is None:
            return False
        tunable_histories = [
            h for name, h in self._histories.items() if self.metrics[name].tunable
        ]
        if not tunable_histories:
            return False
        return all(
            h.rounds_without_improvement() >= self.early_stopping_rounds
            for h in tunable_histories
        )

    def best_values(self):
        """Return {metric_name: best_value} for all tracked metrics."""
        return {name: h.best_value() for name, h in self._histories.items()}

    def best_iterations(self):
        """Return {metric_name: best_iteration} for all tracked metrics."""
        return {name: h.best_iteration() for name, h in self._histories.items()}

    def log_current(self):
        """Log the most recent value and running best for every tracked metric."""
        for name, h in self._histories.items():
            if h.values:
                logging.info(
                    "Metric %s: current=%.6f  best=%.6f  (rounds without improvement: %d)",
                    name,
                    h.values[-1],
                    h.best_value(),
                    h.rounds_without_improvement(),
                )

    def history(self, metric_name):
        """Return the MetricHistory object for a specific metric name."""
        return self._histories[metric_name]

    def summary(self):
        """Return a nested dict with full statistics for all tracked metrics."""
        return {name: h.summary() for name, h in self._histories.items()}

    def xgboost_callback(self):
        """Return an xgboost.callback.TrainingCallback that feeds eval results into this tracker.

        The callback calls record() after each boosting round, logs current values,
        and signals XGBoost to stop early if should_stop_early() returns True.

        Raises ImportError if xgboost is not installed.
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError(
                "xgboost must be installed to use MetricsTracker.xgboost_callback()."
            )

        tracker = self

        class _XGBCallback(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                metric_values = {}
                for dataset, dataset_metrics in evals_log.items():
                    for metric_name, values in dataset_metrics.items():
                        full_name = "{}:{}".format(dataset, metric_name)
                        if values:
                            metric_values[full_name] = values[-1]
                tracker.record(metric_values, iteration=epoch)
                tracker.log_current()
                if tracker.should_stop_early():
                    logging.info(
                        "Early stopping triggered at round %d: no improvement in %d rounds.",
                        epoch + 1,
                        tracker.early_stopping_rounds,
                    )
                    return True
                return False

        return _XGBCallback()
