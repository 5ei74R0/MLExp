"""This module includes squared-relu implementation"""
import torch
from torch import nn
from torch.nn import functional as F


class SquaredReLU(nn.Module):
    def __init__(self, inplace=False):
        super(SquaredReLU, self).__init__()
        self.inplace = inplace

    @torch.jit.script
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(inputs, inplace=self.inplace)
        return x * x

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str
