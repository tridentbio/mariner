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

class TorchgeometricglobaladdpoolArgsTemplate(ApiBaseModel):

class TorchgeometricglobaladdpoolArgs(ApiBaseModel):

class Torchgeometricglobaladdpool(ApiBaseModel):
    type: Literal['torch_geometric.nn.global_add_pool'] = 'torch_geometric.nn.global_add_pool'
    args_template: TorchgeometricglobaladdpoolArgsTemplate

    @classmethod
    def create(cls, args: TorchgeometricglobaladdpoolArgs):
        lib_cls = get_class_from_path_string(cls.type)
        return lib_cls(**args)

class TorchgeometricglobaladdpoolLayer(ApiBaseModel):
    type: Literal['torch_geometric.nn.global_add_pool'] = 'torch_geometric.nn.global_add_pool'
    args: TorchgeometricglobaladdpoolArgs
    input_layer: Optional[bool] = True
    id: str
    forward: str

