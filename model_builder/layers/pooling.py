"""
Pooling layer defined from torch operations
"""

from torch import Tensor, nn


class AddPooling(nn.Module):
    def __init__(self):
        ...

    def forward(self, x: Tensor):
        return x.sum()
