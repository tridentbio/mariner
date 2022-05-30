from enum import Enum
from typing import Any, List, Optional
from pydantic.main import BaseModel
import yaml
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
    input_layer: Optional[bool] = None
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

def get_inputs(x, batch, layer_type: Layer):
    edgeConsumers = [Layer.GINConv2]
    if layer_type == Layer.GlobalAddPool:
        return x, batch.batch
    elif layer_type in edgeConsumers:
        return x, batch.edge_index
    return x

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
                layers_to_callables[layer.name] = nn.Linear(**args)
            elif layer.type == Layer.GINConv2:
                layers_to_callables[layer.name] = layers.GINConvSequentialRelu(**args)
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

    def forward(self, batch):
        current_layer = None
        for layer in self.config.layers:
            if layer.input_layer:
                current_layer = layer
                break
        if current_layer is None:
            raise Exception('No input layer')

        visited = {}
        x = batch.x
        while current_layer and current_layer.output_layer != True:
            visited[current_layer.name] = 1
            print(current_layer.name)
            inputs = get_inputs(x, batch, current_layer.type)
            x = self.layers_to_callables[current_layer.name](*inputs)
            current_layer = self.next_layer(current_layer)
            if current_layer is not None and current_layer.name in visited:
                raise Exception('Cycles not allowed')

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
