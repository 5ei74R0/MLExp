from typing import Dict, Tuple

import pytest
from torch import nn

from mlexp import experiment

call_counter: int = 0


@pytest.mark.parametrize(
    ("epochs"),
    [
        3,
        5,
        10,
    ],
)
def test_run_decorator(epochs):

    global call_counter
    call_counter = 0

    @experiment.run(
        experiment_name="test_exp",
        run_name="test_run",
        params={"param1": 1},
        tags={"tag1": 1},
        epochs=epochs,
    )
    def train_fn_mock() -> Tuple[nn.Module, Dict[str, float], Dict[str, float]]:
        m = nn.Linear(10, 5)
        train_metric_mock1 = 0.1
        train_metric_mock2 = 0.2
        train_metric_mock3 = 0.3
        validation_metric_mock1 = 0.1
        validation_metric_mock2 = 0.2
        validation_metric_mock3 = 0.3
        train_metrics = {
            "train_metric_mock1": train_metric_mock1,
            "train_metric_mock2": train_metric_mock2,
            "train_metric_mock3": train_metric_mock3,
        }
        validation_metrics = {
            "validation_metric_mock1": validation_metric_mock1,
            "validation_metric_mock2": validation_metric_mock2,
            "validation_metric_mock3": validation_metric_mock3,
        }
        global call_counter
        call_counter += 1
        return m, train_metrics, validation_metrics

    train_fn_mock()

    assert call_counter == epochs
