from typing import List, Union

import networkx as nx
import torch
import torch_geometric.nn as geom_nn
from pytorch_lightning.core.lightning import LightningModule
from torch.nn import ReLU, Sigmoid
from torch.optim.adam import Adam

from model_builder.dataset import DataInstance
from model_builder.layers import Concat, GlobalPooling
from model_builder.utils import collect_args

edge_index_classes = geom_nn.MessagePassing
pooling_classes = GlobalPooling

edge_index_classes = geom_nn.MessagePassing
pooling_classes = GlobalPooling
activations = (ReLU, Sigmoid)


def is_message_passing(layer):
    """x = layer(x, edge_index)"""
    return isinstance(layer, geom_nn.MessagePassing)


def is_graph_pooling(layer):
    """x = layer(x, batch)"""
    return isinstance(layer, pooling_classes)


def is_concat_layer(layer):
    return isinstance(layer, Concat)


def is_graph_activation(layer, layers_dict, previous):
    """
    takes the a dictionary with nn.Modules and the keys of
    previous layers, checking if it expectes a pygnn Data instance
    with batch
    """
    if not isinstance(layer, activations):
        return False
    for name in previous:
        if is_message_passing(layers_dict[name]) or is_graph_pooling(layers_dict[name]):
            return True
    return False


def if_str_make_list(x: Union[str, List[str]]) -> List[str]:
    if isinstance(x, str):
        return [x]
    return x


class CustomModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        layers_dict = {}
        self.config = config
        self.layer_configs = {layer.name: layer for layer in config.layers}
        for layer in config.layers:
            layer_instance = layer.create()
            layers_dict[layer.name] = layer_instance
        self._model = torch.nn.ModuleDict(layers_dict)
        self.graph = config.make_graph()
        self.topo_sorting = list(nx.topological_sort(self.graph))
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, input: DataInstance):
        last = input
        for node_name in self.topo_sorting:
            if node_name not in self.layer_configs:
                continue  # is a featurizer, already evaluated by dataset
            layer = self.layer_configs[node_name]
            args = collect_args(input, layer.forward_args.dict())
            if isinstance(args, dict):
                input[layer.name] = self._model[node_name](**args)
            elif isinstance(args, list):
                input[layer.name] = self._model[node_name](*args)
            else:
                input[layer.name] = self._model[node_name](args)
            last = input[layer.name]
        return last

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def test_step(self, batch, batch_idx):
        prediction = self(batch).squeeze()
        loss = self.loss_fn(prediction, batch["y"])
        # self.logger.log('test_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        prediction = self(batch).squeeze()
        loss = self.loss_fn(prediction, batch["y"])
        return loss

    def training_step(self, batch, batch_idx):
        prediction = self(batch).squeeze()
        loss = self.loss_fn(prediction, batch["y"])
        self.log(
            "train_loss", loss, batch_size=len(batch["y"]), on_epoch=True, on_step=False
        )
        return loss
