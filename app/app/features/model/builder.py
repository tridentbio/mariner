from typing import List

import pandas as pd
import torch
from sqlalchemy.orm.session import Session
from torch import nn
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Batch

from app.features.dataset.crud import CRUDDataset
from app.features.model.schema.configs import FeaturizersType, ModelConfig
from app.features.model.schema.layers import LayersType
from app.features.model.utils import get_inputs_from_mask_ltr

# def get_inputs(x, batch, layer_type: Layer):
#    edgeConsumers = [Layer.GINConv2]
#    if layer_type == Layer.GlobalAddPool:
#        return x, batch.batch
#    elif layer_type in edgeConsumers:
#        return x, batch.edge_index
#    return x


# class CustomModel(nn.Module):
#     def __init__(self, config: ModelConfig):
#         super().__init__()
#         self.config = config
#         layers_to_callables = nn.ModuleDict()
#         for layer in config.layers:
#             args = {}
#             if layer.args is not None:
#                 args = layer.args
#             if layer.type == Layer.GlobalAddPool:
#                 layers_to_callables[layer.name] = layers.GlobalAddPool(**args)
#             elif layer.type == Layer.Linear:
#                 layers_to_callables[layer.name] = nn.Linear(**args)
#             elif layer.type == Layer.GINConv2:
#                 layers_to_callables[layer.name] = layers.GINConvSequentialRelu(**args)
#             else:
#                 raise LayerUnknownException(f'No layer named "{layer.type}"')
#         self.layers_to_callables = layers_to_callables
#         # TODO: validate computational graph
#         # Check for cycles, and if inputs leads to outputs
#
#     def next_layer(self, current_layer: LayerConfig) -> Optional[LayerConfig]:
#         for layer in self.config.layers:
#             if layer.name == current_layer.forward:
#                 return layer
#         return None
#
#     def forward(self, batch):
#         current_layer = None
#         for layer in self.config.layers:
#             if layer.input_layer:
#                 current_layer = layer
#                 break
#         if current_layer is None:
#             raise Exception("No input layer")
#
#         visited = {}
#         x = batch.x
#         while current_layer and not current_layer.output_layer:
#             visited[current_layer.name] = 1
#             inputs = get_inputs(x, batch, current_layer.type)
#             x = self.layers_to_callables[current_layer.name](*inputs)
#             current_layer = self.next_layer(current_layer)
#             if current_layer is not None and current_layer.name in visited:
#                 raise Exception("Cycles not allowed")
#
#         if current_layer is None:
#             # Never raised if graph is valid
#             raise Exception("Final layer is not output")
#
#         x = self.layers_to_callables[current_layer.name](x)
#         return x


class CustomModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.callables_by_name = {layer.name: layer.create() for layer in config.layers}
        self.layerconfigs_by_name = {layer.name: layer for layer in config.layers}

    def get_first_layers(self) -> List[str]:
        return []

    def forward(self, batch):
        pass


class CustomGraphModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.callables_by_name = {layer.name: layer.create() for layer in config.layers}
        self.layerconfigs_by_name = {layer.name: layer for layer in config.layers}

    def get_featurizer_layer(self) -> LayersType:
        name = self.config.featurizer.forward
        return self.layerconfigs_by_name[name]

    def forward(self, batch):
        batch = batch
        print(batch)
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch
        print("from dataset: ", x, edge_index, batch)
        current_layer = self.get_featurizer_layer()
        while not current_layer.is_output_layer:
            layercallable = self.callables_by_name[current_layer.name]
            print(current_layer)
            print("mask :", format(current_layer.forward_input_mask, "b"))
            inputs = get_inputs_from_mask_ltr(
                [x, edge_index, batch], current_layer.forward_input_mask
            )
            print(inputs)
            x = layercallable(*inputs)

        inputs = get_inputs_from_mask_ltr(
            [x, edge_index, batch], current_layer.forward_input_mask
        )
        x = self.callables_by_name[current_layer.name](x)
        return x


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
