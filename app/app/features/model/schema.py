from typing import Any, Optional, Literal, Union
from app.schemas.api import ApiBaseModel


class CycleInGraphException(ApiBaseModel):
    """
    Raised when cycles are detected in the computational
    graph of the model
    """
    code = 200
    message = "There is a cycle in the graph"


class Featurizer(ApiBaseModel):
    name: str
    args: Any

def get_module_name(classpath: str) -> str:
    return '.'.join(classpath.split('.')[:-1])

class LinearArgsTemplate(ApiBaseModel):
    in_features: Literal["int"] = "int"
    out_features: Literal["int"] = "int"

class LinearArgs(ApiBaseModel):
    in_features: int
    out_features: int

class Linear(ApiBaseModel):
    type: Literal['torch.nn.Linear'] = 'torch.nn.Linear'
    args_template: LinearArgsTemplate

    @classmethod
    def create(cls, args: LinearArgs):
        module_name = get_module_name(cls.type)
        eval(f'import {module_name}')
        lib_cls = eval('{self.type}')
        return lib_cls(**args)

class LinearLayer(ApiBaseModel):
    type: Literal['torch.nn.Linear'] = 'torch.nn.Linear'
    args: LinearArgs
    input_layer: Optional[bool] = True
    id: str
    forward: str

class GCNConvArgsTemplate(ApiBaseModel):
    in_channels: Literal["int"] = "int"
    out_channels: Literal["int"] = "int"

class GCNConvArgs(ApiBaseModel):
    in_channels: int
    out_channels: int

class GCNConv(ApiBaseModel):
    type: Literal['torch_geomtric.nn.GINConv'] = 'torch_geomtric.nn.GINConv'
    args_template: GCNConvArgsTemplate

    @classmethod
    def create(cls, args: LinearArgs):
        module_name = get_module_name(cls.type)
        eval(f'import {module_name}')
        lib_cls = eval('{self.type}')
        return lib_cls(**args)

class GCNConvLayer(ApiBaseModel):
    type: Literal['torch.nn.Linear'] = 'torch.nn.Linear'
    args: LinearArgs
    input_layer: Optional[bool] = True
    id: str
    forward: str


layer_args = LinearArgs(in_features=10, out_features=64)
layer = Linear.create(args=layer_args) 
# equilavent to:
# >>> torch.nn.Linear(in_features=10, out_features=64)


class Model(ApiBaseModel):
    name: str
    input_shape: ... # TODO: we probly need a dataset
    output_shape: ... # TODO: We probly need target columns
    featurizer: Featurizer
    layers: Union[Linear, GCNConv]


