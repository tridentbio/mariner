"""
Pooling layer defined from torch operations
"""

from typing import Union

from torch import Tensor, nn


class AddPooling(nn.Module):
    """
    Returns the summation of it's inputs (using tensor.sum(dim=dim))

    See:
        <https://pytorch.org/docs/stable/generated/torch.sum.html>
    """

    def __init__(self, dim: Union[int, None] = None):
        nn.Module.__init__(self)
        self.dim = dim

    def forward(self, x: Tensor):
        """Returns summation of tensor items

        Args:
            x: torch.Tensor
        """
        return x.sum(dim=self.dim)
