import torch
from torch import nn
from torch.nn import functional as F


class OneHot(nn.Module):
    """
    A helper layer that outputs the one-hot encoding representation of
    it's categorical inputs
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        super().__init__()

    def forward(self, x1: torch.Tensor):
        return F.one_hot(x1, num_classes=self.num_classes)
