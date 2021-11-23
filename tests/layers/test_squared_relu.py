import pytest
import torch

from mlexp.layers.activation import SquaredReLU


@pytest.mark.parametrize(
    (
        "inputs",
        "expected",
    ),
    [
        (
            torch.Tensor([-1, -0.6, 3, 0.5]),
            torch.Tensor([0, 0, 9, 0.25]),
        ),
        (
            torch.Tensor([[-1, -2, 3], [4, -4, -6], [14, -16, 0]]),
            torch.Tensor([[0, 0, 9], [16, 0, 0], [196, 0, 0]]),
        ),
    ],
)
def test_squared_relu(inputs: torch.Tensor, expected: torch.Tensor):
    # tests forward
    squared_relu = SquaredReLU()
    outputs = squared_relu(inputs)
    assert torch.equal(outputs, expected)

    # tests whether or not autograd finishes successfully
    o_size = 3
    fc = torch.nn.Linear(outputs.size()[0], o_size)
    outputs = squared_relu(fc(outputs))
    lsfn = torch.nn.MSELoss()
    loss = lsfn(outputs, torch.zeros(outputs.size()))
    loss.backward()
