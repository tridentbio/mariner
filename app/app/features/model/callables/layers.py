import torch
import torch_geometric.nn as pygnn
from torch import nn
from torch_geometric.nn import global_add_pool


class GlobalAddPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.func = global_add_pool

    def forward(self, x, edge_list):
        return self.func(x, edge_list)


class GINConvSequentialRelu(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
        )
        self.layer = pygnn.GINConv(mlp, train_eps=True)

    def forward(self, x: torch.Tensor, edge_index):
        return self.layer(x, edge_index)
