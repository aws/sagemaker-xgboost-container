from sklearn.metrics import accuracy_score, f1_score, mean_squared_error


# TODO: Rename both according to AutoML standards
def accuracy(preds, dtrain):
    """Compute accuracy.

    :param preds: Prediction values
    :param dtrain: Training data with labels
    :return: Metric name, accuracy value.
    """
    labels = dtrain.get_label()
    y_bin = [1. if preds_cont > 0.5 else 0. for preds_cont in preds]  # binaryzing output
    return 'accuracy', accuracy_score(labels, y_bin)


def f1(preds, dtrain):
    """Compute f1 score. This can be used for multiclassification training.

    For more information see: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

    :param preds: Prediction values
    :param dtrain: Training data with labels
    :return: Metric name, f1 score
    """
    labels = dtrain.get_label()
    y_bin = [1. if preds_cont > 0.5 else 0. for preds_cont in preds]  # binaryzing output
    return 'f1', f1_score(labels, y_bin)


def mse(preds, dtrain):
    """Compute mean squared error.

    For more information see: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    :param preds: Prediction values
    :param dtrain: Training data with labels
    :return: Metric name, mean squared error
    """
    labels = dtrain.get_label()
    return 'mse', mean_squared_error(labels, preds)


CUSTOM_METRICS = {
    "accuracy": accuracy,
    "f1": f1,
    "mse": mse
}


def get_custom_metrics(eval_metrics):
    """Get container defined metrics from metrics list."""
    return set(eval_metrics).intersection(CUSTOM_METRICS.keys())


def configure_feval(custom_metric_list):
    """Configure custom_feval method with metrics specified by user.

    XGBoost.train() can take a feval argument whose value is a function. This method configures that function with
    multipl metrics if required, then returns to use during training.

    :param custom_metric_list: Metrics to evaluate using feval
    :return: Configured feval method
    """
    def custom_feval(preds, dtrain):
        metrics = []

        for metric_method_name in custom_metric_list:
            custom_metric = CUSTOM_METRICS[metric_method_name]
            metrics.append(custom_metric(preds, dtrain))

        return metrics

    return custom_feval
