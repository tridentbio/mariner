from enum import Enum
from typing import Any, List, Optional
from pydantic.main import BaseModel
import yaml
import torch
from torch import nn
from ast import literal_eval

from app.features.model.callables import layers

class Tuple(str):
    val: tuple

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def _modify_schema__(cls, field_schema):
        field_schema.update(examples=["(1,)", "(1,1,0)"])

    @classmethod
    def validate(cls, v):
        try:
            t = literal_eval(v)
            if not isinstance(t, tuple):
                raise ValueError("Tuple(s), s should evaluate to a tuple")
            return cls(v)
        except Exception:
            raise

class Featurizer(Enum):
    MoleculeFeaturizer = 'MoleculeFeaturizer'

class Layer(Enum):
    GINConv2 = 'GINConv2'
    GlobalAddPool = 'torch_geometric.global_add_pool'
    Linear = 'Linear'

class FeaturizerConfig(BaseModel):
    name: Featurizer
    args: Any

class LayerConfig(BaseModel):
    name: str
    initial_layer: Optional[bool] = None
    output_layer: Optional[bool] = None
    type: Layer
    args: Any
    forward: Optional[str] = None

class ModelConfig(BaseModel):
    name: str
    input_shape: Tuple
    output_shape: Tuple
    featurizer: FeaturizerConfig
    layers: List[LayerConfig]


class LayerUnknownException(Exception):
    pass

class CustomModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        layers_to_callables = {}
        for layer in config.layers:
            args = {}
            if layer.args is not None:
                args = layer.args
            if layer.type == Layer.GlobalAddPool:
                layers_to_callables[layer.name] = layers.GlobalAddPool(**args)
            elif layer.type == Layer.Linear:
                layers_to_callables[layer.type] = nn.Linear(**args)
            elif layer.type == Layer.GINConv2:
                layers_to_callables[layer.type] = layers.GINConvSequentialRelu(**args)
            else:
                raise LayerUnknownException(f'No layer named "{layer.type}"')
        self.layers_to_callables = layers_to_callables
        # TODO: validate computational graph
        # Check for cycles, and if inputs leads to outputs

    def next_layer(self, current_layer: LayerConfig) -> Optional[LayerConfig]:
        for layer in self.config.layers:
            if layer.name == current_layer.forward:
                return layer
        return None

    def forward(self, x: torch.Tensor):
        current_layer = None
        for layer in self.config.layers:
            if layer.initial_layer:
                current_layer = layer
                break
        if current_layer is None:
            raise Exception('No first layer')

        while current_layer and current_layer.output_layer != True:
            x = self.layers_to_callables[current_layer.name](x)
            current_layer = self.next_layer(current_layer)

        if current_layer is None:
            # Never raised if graph is valid
            raise Exception("Final layer is not output")

        x = self.layers_to_callables[current_layer.name](x)
        return x


def build_model_from_yaml(yamlstr: str) -> nn.Module:
    config_dict = yaml.safe_load(yamlstr)
    config = ModelConfig.parse_obj(config_dict)
    model = CustomModel(config)
    return model
tqq
