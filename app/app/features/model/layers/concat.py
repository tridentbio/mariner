import torch
from torch import nn


class Concat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        return torch.cat([x1, x2], dim=-1)
