from typing import List, Union

import networkx as nx
import torch
import torch_geometric.nn as geom_nn
from pytorch_lightning.core.lightning import LightningModule
from torch.nn import ReLU, Sigmoid
from torch.optim.adam import Adam
from torch_geometric.data.data import Data

from app.features.model.layers import Concat, GlobalPooling
from app.features.model.schema.configs import ModelConfig

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
    previous layers, checking if
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
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        layers_dict = {}
        for layer in config.layers:
            layers_dict[layer.name] = layer.create()

        self.layers = torch.nn.ModuleDict(layers_dict)

        self.layer_configs = {l.name: l for l in config.layers}

        self.graph = config.make_graph()
        self.topo_sorting = list(nx.topological_sort(self.graph))

        self.loss_fn = torch.nn.MSELoss()

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

    def forward(self, input_):
        storage = input_.copy()

        for key, value in storage.items():
            if key != "y" and not isinstance(value, Data):
                storage[key] = value.reshape(len(value), 1)

        batch_values = None
        last = None

        for index, node_name in enumerate(self.topo_sorting):
            if not node_name in self.layers:
                continue
            layer_name = node_name
            layer = self.layers[layer_name]
            layer_config = self.layer_configs[layer_name]
            previous_layers = [
                p_layer for p_layer, c_layer in self.graph.in_edges(layer_name)
            ]
            inputs = if_str_make_list(layer_config.input)
            # Step 2
            # Transform and preprocess the input and output based on the previous
            # and next layers.

            if is_message_passing(layer):
                assert (
                    len(inputs) == 1
                ), f"Length of a gnn layer's inputs should be at most 1. inputs = {inputs}"
                src = inputs[0]
                assert isinstance(storage[src], Data)
                x, edge_index, batch_values = (
                    storage[src].x,
                    storage[src].edge_index,
                    storage[src].batch,
                )
                x, edge_index = layer(x=x, edge_index=edge_index), edge_index
                storage[layer_name] = Data(
                    x=x, edge_index=edge_index, batch=batch_values
                )
            # 2.2   - if its an pooling layer it always have just one input
            #         and the input is composed by x (node_features) from the last layer
            #         and the batch that comes with the data object
            elif is_graph_pooling(layer):
                assert (
                    len(inputs) == 1
                ), f"Length of a gnn layer's inputs should be at most 1. inputs = {inputs}"
                src = inputs[0]
                assert isinstance(storage[src], Data)
                x, edge_index, batch_values = (
                    storage[src].x,
                    storage[src].edge_index,
                    storage[src].batch,
                )
                storage[layer_name] = layer(x=x, batch=batch_values)
            # 2.3   - if its an activation or a mlp/normal layer we need to check the
            #         previous layers to make sure that the input is in t;he correct format
            #         and check the next layers to format the output
            elif is_graph_activation(layer, self.layers, previous_layers):
                assert (
                    len(inputs) == 1
                ), f"Length of a activation layer's inputs should be at most 1. inputs = {inputs}"
                src = inputs[0]
                assert isinstance(storage[src], Data)
                x, edge_index, batch_values = (
                    storage[src].x,
                    storage[src].edge_index,
                    storage[src].batch,
                )
                storage[layer_name] = Data(
                    x=layer(x), edge_index=edge_index, batch=batch_values
                )
            elif is_concat_layer(layer):
                assert (
                    len(inputs) == 2
                ), f"Length of a concat layer's inputs should be 2. inputs = {inputs}"
                x1, x2 = storage[inputs[0]], storage[inputs[1]]
                storage[layer_name] = layer(x1, x2)
            elif isinstance(layer, torch.nn.Linear):
                input_values = [storage[input] for input in inputs]
                input_values = torch.cat(input_values, dim=1)
                x_ = layer(input_values)
                storage[layer_name] = x_
            else:
                input_values = [
                    storage[input]
                    if isinstance(storage[input], Data)
                    else storage[input]
                    for input in inputs
                ]
                x_ = layer(*input_values)
                storage[layer_name] = x_
            last = storage[layer_name]
        return last
