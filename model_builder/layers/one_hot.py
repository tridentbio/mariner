from typing import Optional, Union

import torch
from torch import nn
from torch.nn import functional as F


class OneHot(nn.Module):
    """
    A helper layer that outputs the one-hot encoding representation of
    it's categorical inputs
    """

    # this property is only filled on training, when we have the dataset "at hands"
    classes: Optional[dict[Union[str, int], int]] = None

    def __init__(self):
        super().__init__()

    def forward(self, x1: Union[list[str], list[int]]):
        assert self.classes, "OneHot layer is missing the classes property set"
        longs = torch.Tensor([self.classes[x] for x in x1]).long()
        return F.one_hot(longs, num_classes=len(self.classes)).float()
