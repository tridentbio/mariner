from typing import Dict, List, Union

import networkx as nx
import torch
import torch_geometric.nn as geom_nn
from pandas.core.frame import DataFrame
from pytorch_lightning import LightningModule
from sqlalchemy.orm.session import Session
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Dataset as PygDataset
from torch_geometric.data.data import Data

from app.features.dataset.crud import CRUDDataset
from app.features.model.layers import Concat, GlobalPooling
from app.features.model.schema.configs import ModelConfig

# detour. how to check the forward signature?
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


CustomDatasetIn = Dict[str, Union[torch.Tensor, Data]]


class CustomModel(LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        layers_dict = {}
        for layer in config.layers:
            layers_dict[layer.name] = layer.create()

        self.layers = torch.nn.ModuleDict(layers_dict)

        self.layer_configs = {layer.name: layer for layer in config.layers}

        self.graph = config.make_graph()
        self.topo_sorting = list(nx.topological_sort(self.graph))

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # TODO: adapt loss to problem type. MSE will only work for regression
        loss = F.mse_loss(logits, y)
        return loss

    def forward(self, input_: CustomDatasetIn):
        storage = input_.copy()

        for index, node_name in enumerate(self.topo_sorting):
            if node_name not in self.layers:
                continue
            layer_name = node_name
            layer = self.layers[layer_name]
            layer_config = self.layer_configs[layer_name]
            previous_layers = [
                p_layer for p_layer, c_layer in self.graph.in_edges(layer_name)
            ]
            inputs = if_str_make_list(layer_config.input)

            if is_message_passing(layer):
                assert (
                    len(inputs) == 1
                ), "Length of a gnn layer's inputs should be at most 1. "
                f"inputs = {inputs}"
                src = inputs[0]
                assert isinstance(storage[src], Data)
                x, edge_index = storage[src].x, storage[src].edge_index
                x, edge_index = layer(x=x, edge_index=edge_index), edge_index
                storage[layer_name] = Data(x=x, edge_index=edge_index)
            elif is_graph_pooling(layer):
                assert (
                    len(inputs) == 1
                ), "Length of a gnn layer's inputs should be at most 1. "
                f"inputs = {inputs}"
                src = inputs[0]
                assert isinstance(storage[src], Data)
                x, edge_index, batch = (
                    storage[src].x,
                    storage[src].edge_index,
                    storage[src].batch,
                )
                storage[layer_name] = layer(x=x, batch=batch)
            elif is_graph_activation(layer, self.layers, previous_layers):
                assert (
                    len(inputs) == 1
                ), "Length of a activation layer's inputs should be at most 1. "
                f"inputs = {inputs}"
                src = inputs[0]
                assert isinstance(storage[src], Data)
                x, edge_index = storage[src].x, storage[src].edge_index
                storage[layer_name] = Data(x=layer(x), edge_index=edge_index)
            elif is_concat_layer(layer):
                assert (
                    len(inputs) == 2
                ), f"Length of a concat layer's inputs should be 2. inputs = {inputs}"
                x1, x2 = storage[inputs[0]], storage[inputs[1]]
                storage[layer_name] = layer(x1, x2)
            else:
                input_values = [
                    storage[input]
                    if isinstance(storage[input], Data)
                    else storage[input]
                    for input in inputs
                ]
                storage[layer_name] = layer(*input_values)
            last = storage[layer_name]
        return last


def build_model_from_yaml(yamlstr: str) -> nn.Module:
    config = ModelConfig.from_yaml(yamlstr)
    model = CustomModel(config)
    return model


def if_str_make_list(str_or_list: Union[str, List[str]]) -> List[str]:
    if isinstance(str_or_list, str):
        return [str_or_list]
    return str_or_list


class CustomDataset(PygDataset):
    def __init__(self, dataset: DataFrame, model_config: ModelConfig):
        super().__init__()
        self.dataset = dataset
        self.model_config = model_config
        self.df = dataset
        # maps featurizer name to actual featurizer instance
        self.featurizers_dict = {f.name: f.create() for f in model_config.featurizers}
        # maps featurizer name to featurizer config
        self.featurizer_configs = {f.name: f for f in model_config.featurizers}

    def __len__(self):
        return len(self.df)

    def get_not_featurized(self):
        is_col_featurized = {
            col: False for col in self.model_config.dataset.feature_columns
        }
        for feat_config in self.featurizer_configs.values():
            inputs = if_str_make_list(feat_config.input)
            for input in inputs:
                is_col_featurized[input] = True
        not_featurized = [
            col for col, is_featurized in is_col_featurized.items() if not is_featurized
        ]
        return not_featurized

    def __getitem__(self, idx):
        sample = {}
        target_column = self.model_config.dataset.target_column
        y = torch.Tensor([self.df.loc[idx, [target_column]]])
        not_featurized_cols = self.get_not_featurized()
        for column in not_featurized_cols:
            # TODO, validate columns. For now assuming are all scalars
            sample[column] = torch.Tensor([self.df.loc[idx, column]])
        for feat_name, feat in self.featurizers_dict.items():
            feat_config = self.featurizer_configs[feat_name]
            inputs = if_str_make_list(feat_config.input)
            assert len(inputs) == 1, "Only featurizers with a single input for now"
            sample[feat_name] = feat(self.df.loc[idx, inputs[0]])

        return sample, y


def build_dataset(
    model_config: ModelConfig,
    dataset_repo: CRUDDataset,
    db: Session,
) -> TorchDataset:
    dataset = dataset_repo.get_by_name(db, model_config.dataset.name)
    df = dataset.get_dataframe()
    return CustomDataset(df, model_config)
