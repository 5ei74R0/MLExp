"""This module includes squared-relu implementation"""
import torch
from torch import nn
from torch.nn import functional as F


class SquaredReLU(nn.Module):
    """
    Squared ReLU layer. proposed in "Primer: Searching for Efficient Transformers for Language Modeling"\\
    This Layer is implemented by using `F.relu` directly and goes fast.
    """

    def __init__(self, inplace=True):
        super(SquaredReLU, self).__init__()
        self.inplace = inplace

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(inputs, inplace=self.inplace)
        return x * x

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str
