import torch
from torch import nn
from torch_geometric.nn import GINConv, global_add_pool


class GlobalAddPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.func = global_add_pool

    def forward(self, x, edge_list):
        return self.func(x, edge_list)


class ExampleModel(nn.Module):
    def __init__(
        self,
        num_layers: int = 4,
        in_channels: int = 30,
        hidden_channels: int = 64,
        out_features: int = 1,
    ):
        super().__init__()
        layers = []

        self.pool_function = GlobalAddPool()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )

            layers.append(GINConv(mlp, train_eps=True))
            in_channels = hidden_channels

        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(
            in_features=hidden_channels, out_features=out_features
        )

    def forward(self, batch):
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch

        for layer in self.layers:
            x = layer(x, edge_index)

        pred = self.pool_function(x, batch)
        pred = self.out(pred)

        return pred


if __name__ == "__main__":
    model = ExampleModel()
    torch.save(model, "app/tests/data/model.pt")
