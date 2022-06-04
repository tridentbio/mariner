from typing import Any, Optional, Literal, Union
from app.schemas.api import ApiBaseModel

def get_module_name(classpath: str) -> str:
    return '.'.join(classpath.split('.')[:-1])

def get_class_from_path_string(pathstring: str):
    module_name = get_module_name(pathstring)
    code = f'''
import {module_name}
references['cls'] = {pathstring}
'''
    references = {} # cls must be a reference
    exec(code, globals(), { 'references': references })
    return references['cls']

class TorchgeometricgcnconvArgsTemplate(ApiBaseModel):
    in_channels: Literal["int"] = "int"
    out_channels: Literal["int"] = "int"

class TorchgeometricgcnconvArgs(ApiBaseModel):
  in_channels: int
  out_channels: int

class Torchgeometricgcnconv(ApiBaseModel):
    type: Literal['torch_geometric.nn.GCNConv'] = 'torch_geometric.nn.GCNConv'
    args_template: TorchgeometricgcnconvArgsTemplate

    @classmethod
    def create(cls, args: TorchgeometricgcnconvArgs):
        lib_cls = get_class_from_path_string(cls.type)
        return lib_cls(**args)

class TorchgeometricgcnconvLayer(ApiBaseModel):
    type: Literal['torch_geometric.nn.GCNConv'] = 'torch_geometric.nn.GCNConv'
    args: TorchgeometricgcnconvArgs
    input_layer: Optional[bool] = True
    id: str
    forward: str

