import functools
from typing import Any, Callable, Dict

import mlflow


class run:
    """
    decorator that decorate an function for training and validation loop
    in order to repetitively run the loop and observe metrics in every epoch with mlflow
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        params: Dict[str, Any],
        tags: Dict[str, Any],
        epochs: int,
    ):
        """
        decorate an training and validation loop
        in order to repetitively run the loop
        and observe metrics in every epoch with mlflow

        Args:
            experiment_name (str): Used as `mlflow.set_experiment()`
            run_name (str): Used as `mlflow.start_run(run_name)`
            params (Dict[str, Any]): Used as `mlflow.log_params(params)`
            tags (Dict[str, Any]): Used as `mlflow.set_tags(tags)`
            epochs (int): Number of epochs to run training and validation loop
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.params = params
        self.tags = tags
        self.epochs = epochs

    def __call__(self, loop_fn: Callable):
        @functools.wraps(loop_fn)
        def wrapper_fn(*args, **kwargs):
            mlflow.set_experiment(self.experiment_name)
            with mlflow.start_run(run_name=self.run_name):
                mlflow.log_params(self.params)
                mlflow.set_tags(self.tags)
                for step in range(1, self.epochs + 1):
                    model, train_metrics, validation_metrics = loop_fn(*args, **kwargs)
                    self._register_metrics(prefix="train", metrics=train_metrics, step=step)
                    self._register_metrics(prefix="validation", metrics=validation_metrics, step=step)
            return model, train_metrics, validation_metrics

        return wrapper_fn

    def _register_metrics(self, prefix: str, metrics: Dict[str, float], step: int):
        for metric, value in metrics.items():
            mlflow.log_metric(key=prefix + " " + metric, value=value, step=step)
