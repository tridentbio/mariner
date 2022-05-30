from typing import Optional
import pandas as pd
from pandas.core.frame import DataFrame
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch_geometric.nn as pygnn
from torch_geometric.nn import global_add_pool

from app.features.model.featurizers import MoleculeFeaturizer

class ExampleDataset(Dataset):
    
    def __init__(self, path: str, target: Optional[str] = None, **kwargs):
        self.feat = MoleculeFeaturizer(**kwargs)
        self.data: DataFrame = DataFrame(pd.read_csv(path))
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = dict(self.data.iloc[idx, :])
        graph = self.feat(sample['smiles'])

        if self.target:
            graph.y = torch.tensor([sample[self.target]])
        
        return graph


class GlobalAddPool(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.func = global_add_pool
        
    def forward(self, x, edge_list):
        return self.func(x, edge_list)

class ExampleModel(nn.Module):
    
    def __init__(self, num_layers: int = 4, in_channels: int = 30, hidden_channels: int = 64, out_channels: int = 1, **kwargs):
        super().__init__()
        layers = []
        
        self.pool_function = GlobalAddPool()
        
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            
            layers.append(pygnn.GINConv(mlp, train_eps=True))
            in_channels = hidden_channels

        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(in_features=hidden_channels, out_features=1)
        
    def forward(self, batch):
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch
        
        for layer in self.layers:
            x = layer(x, edge_index)
            
        pred = self.pool_function(x, batch)
        pred = self.out(pred)
        
        return pred

def create_example_model():
    return ExampleModel()
