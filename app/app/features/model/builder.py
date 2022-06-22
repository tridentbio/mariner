from typing import Dict, List, Union

import networkx as nx
import pandas as pd
import torch
import torch_geometric.nn as geom_nn
from sqlalchemy.orm.session import Session
from torch import nn
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Batch
from torch_geometric.data.data import Data

from app.features.dataset.crud import CRUDDataset
from app.features.model.layers import Concat, GlobalPooling
from app.features.model.schema.configs import FeaturizersType, ModelConfig

# detour. how to check the forward signature?
edge_index_classes = geom_nn.MessagePassing
pooling_classes = GlobalPooling


def is_message_passing(layer):
    """x = layer(x, edge_index)"""
    return isinstance(layer, geom_nn.MessagePassing)


def is_graph_pooling(layer):
    """x = layer(x, batch)"""
    return isinstance(layer, pooling_classes)


def is_concat_layer(layer):
    return isinstance(layer, Concat)


CustomDatasetIn = Dict[str, Union[torch.Tensor, Data]]


class CustomModel(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        layers_dict = {}
        for layer in config.layers:
            layers_dict[layer.name] = layer.create()
        self.layers = torch.nn.ModuleDict(layers_dict)
        self.layer_configs = {layer.type: layer for layer in config.layers}
        graph = config.make_graph()
        self.topo_sorting = nx.topological_sort(graph)

    def forward(self, x: CustomDatasetIn):
        outs = x.copy()
        for layer_name in self.topo_sorting:
            layer = self.layers[layer_name]
            inputs = self.layer_configs[layer_name].input

            if is_message_passing(layer):
                x, edge_index = inputs.x, inputs.edge_index
                outs[layer_name] = layer(x, edge_index)
            elif is_graph_pooling(layer):
                x, batch = inputs.x, inputs.batch
                outs[layer_name] = layer(x, batch)
            else:  # concat layers and normal layers
                if isinstance(inputs, str):  # arrays only
                    inputs = [inputs]
                inputs = [outs[input_name] for input_name in inputs]
                outs[layer_name] = layer(*inputs)
            last = outs[layer_name]
        return last


def build_model_from_yaml(yamlstr: str) -> nn.Module:
    config = ModelConfig.from_yaml(yamlstr)
    model = CustomModel(config)
    return model


class BuilderDataset(TorchDataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_col: str,
        feature_columns: List[str],
        featurizers: List[FeaturizersType],
    ):
        self.target_col = target_col
        self.feature_columns = feature_columns
        self.data = dataframe
        self.featurizer_configs = {
            featurizer.name: featurizer for featurizer in featurizers
        }
        self.featurizers = {
            featurizer.name: featurizer.create() for featurizer in featurizers
        }
        self.callables = {}

    def __len__(self):
        return len(self.data)

    def split_featurized_and_not(self):
        featurized = {col: False for col in self.feature_columns}
        for _, feat in self.featurizer_configs.items():
            for col in feat.column_names:
                featurized[col] = True
        featurized_cols = {col: True for col in featurized if featurized[col]}.keys()
        unfeaturized_cols = {
            col: True for col in featurized if not featurized[col]
        }.keys()

        return featurized_cols, unfeaturized_cols

    def __getitem__(self, idx: int):
        sample = dict(self.data.iloc[idx, :])
        _, not_featurized_columns = self.split_featurized_and_not()
        out = {}
        for feat_name, feat in self.featurizers.items():
            [col] = self.featurizer_configs[feat_name].column_names
            graph = feat(sample[col])
            out[feat_name] = graph  # torch_geometric Data class
        batch = Batch([g for _, g in out.items()])
        batch.featurizers = [feat for _, feat in self.featurizers.items()]
        batch.not_featurized_columns = not_featurized_columns
        batch.not_featurized = self.data.loc[idx, not_featurized_columns]
        batch.y = torch.Tensor([sample[self.target_col]])
        return batch


def build_dataset(
    model_config: ModelConfig,
    dataset_repo: CRUDDataset,
    db: Session,
) -> TorchDataset:
    dataset = dataset_repo.get_by_name(db, model_config.dataset.name)
    ds_config = model_config.dataset
    df = dataset.get_dataframe()
    featurizers = [model_config.featurizer]
    return BuilderDataset(
        df,
        target_col=ds_config.target_column,
        feature_columns=ds_config.feature_columns,
        featurizers=featurizers,
    )
