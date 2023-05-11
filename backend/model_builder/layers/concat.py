"""
Concat layer definition
"""
from typing import List

import torch
from torch import nn


class Concat(nn.Module):
    """
    A helper layer that concatenates the outputs of 2 other layers.

    Attributes:
        dim: dimension to use on concatenation.
    """

    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        """Performs a concatenation as done by ``torch.cat``.

        Args:
            xs: List of inputs to be concatenated on ``dim`` dimension.

        Returns:
            A tensor with the concatenated inputs.
        """
        return torch.cat(xs, dim=self.dim)


Concat.__doc__ = "(Based on torch.cat)\n" + (torch.cat.__doc__ or "")