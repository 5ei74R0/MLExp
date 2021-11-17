from typing import Tuple

import pytest
from torch import nn

from mlexp import experiment
from mlexp.experiment.runner import metrics_t

# global param
call_counter: int = 0

# mock metrics
most_simple_metrics: metrics_t = 0.3
general_metrics: metrics_t = {
    "train": {
        "acc": 0.56,
        "loss": 1.6489,
    },
    "validation": {
        "acc": 0.49,
        "loss": 1.8979,
    },
}
complicated_metrics: metrics_t = {
    "train": {
        "acc": {
            "model_a": 0.30,
            "model_b": 0.56,
            "model_c": {"case1": 0.43, "case2": 0.46},
        },
        "loss": 1.6489,
    },
    "validation": {
        "average acc": 0.49,
        "loss": 1.8979,
    },
}


@pytest.mark.parametrize(
    ("epochs", "metrics"),
    [
        (3, most_simple_metrics),  # case1. type(metrics) == float
        (5, general_metrics),
        (10, complicated_metrics),
    ],
)
def test_run_decorator(epochs, metrics):

    global call_counter
    call_counter = 0

    @experiment.run(
        experiment_name="test_exp",
        run_name="test_run",
        params={"param1": 1},
        tags={"tag1": 1},
        epochs=epochs,
    )
    def train_fn_mock(mtrc: metrics_t) -> Tuple[nn.Module, metrics_t]:
        m = nn.Linear(10, 5)
        global call_counter
        call_counter += 1
        return m, mtrc

    train_fn_mock(metrics)

    assert call_counter == epochs
