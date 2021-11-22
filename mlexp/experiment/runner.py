import functools
from typing import Any, Callable, Mapping, Tuple

import mlflow

from mlexp._types import nested_dict_t

metrics_t = nested_dict_t[float]


class run:
    """
    decorator that decorates an function for training and validation loop
    in order to repetitively run the loop and observe metrics in every epoch with mlflow

    Example:
        target function should return the `tuple[object, metrics_t]`\\
        where `metrics_t = Mapping[str, metrics_t] | float`\\
        it means metrics_t could be `float`, `dict[str, float]`,\\
        and more complicated recursive nest could also be accommodated\\
        such as `dict[dict[..[dict[str, float], float]]]`\\
        and you can put anything in the first element
        (In most cases, you may put your trained model here)\\
        for instance, the following code is available::

            @mlexp.experiment.run(
                exp1,
                running1,
                params={
                    "p1": 1,
                },
                tags={
                    "tag1": 1,
                },
                epochs=10,
            )
            def fn():
                metrics = {
                    "train": {
                        "acc": {
                            "model_a": 0.30,
                            "model_b": 0.56,
                            "average": 0.43,
                        },
                        "loss": 1.6489,
                    },
                    "validation": {
                        "average acc": 0.49,
                        "loss": 1.8979,
                    },
                }
                return 1, metrics

    """

    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        params: nested_dict_t,
        tags: nested_dict_t,
        epochs: int,
        prefix_connector: str = "-",
    ):
        """
        decorate an training and validation loop
        in order to repetitively run the loop
        and observe metrics in every epoch with mlflow

        Args:
            experiment_name (`str`): Used as `mlflow.set_experiment()`
            run_name (`str`): Used as `mlflow.start_run(run_name)`
            params (`nested_dict_t`): Registered as parameters in a run
            tags (`nested_dict_t`): Registered as tags in a run
            epochs (`int`): Number of epochs to run training and validation loop
            prefix_connector (`str`): Connects parent-prefix & child-prefix in nested_dict
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.params = params
        self.tags = tags
        self.epochs = epochs
        self.prefix_connector = prefix_connector

    def __call__(self, loop_fn: Callable[..., Tuple[Any, metrics_t]]):
        @functools.wraps(loop_fn)
        def wrapper_fn(*args, **kwargs):
            mlflow.set_experiment(self.experiment_name)
            with mlflow.start_run(run_name=self.run_name):
                self._recursive_params_logging(self.params)
                self._recursive_tags_setting(self.tags)

                # run epochs
                for step in range(1, self.epochs + 1):
                    model, metrics = loop_fn(*args, **kwargs)
                    self._recursive_metrics_logging(step, metrics)
            return model, metrics

        return wrapper_fn

    def _recursive_metrics_logging(
        self,
        step: int,
        metrics: metrics_t,
        prefix: str = "",
    ) -> None:
        if not isinstance(metrics, Mapping):
            if prefix == "":
                prefix = "NO-NAME"
            mlflow.log_metric(key=prefix, value=metrics, step=step)
            return

        for key, sub_metrics in metrics.items():
            n_prefix = prefix + self.prefix_connector + key if prefix != "" else key
            self._recursive_metrics_logging(step, sub_metrics, prefix=n_prefix)

    def _recursive_params_logging(
        self,
        params: nested_dict_t,
        prefix: str = "",
    ) -> None:
        if not isinstance(params, Mapping):
            if prefix == "":
                prefix = "NO-NAME"
            mlflow.log_param(key=prefix, value=params)
            return

        for key, sub_params in params.items():
            n_prefix = prefix + self.prefix_connector + key if prefix != "" else key
            self._recursive_params_logging(sub_params, prefix=n_prefix)

    def _recursive_tags_setting(
        self,
        tags: nested_dict_t,
        prefix: str = "",
    ) -> None:
        if not isinstance(tags, Mapping):
            if prefix == "":
                prefix = "NO-NAME"
            mlflow.set_tag(key=prefix, value=tags)
            return

        for key, sub_params in tags.items():
            n_prefix = prefix + self.prefix_connector + key if prefix != "" else key
            self._recursive_tags_setting(sub_params, prefix=n_prefix)
