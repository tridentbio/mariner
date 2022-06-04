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

class TorchlinearArgsTemplate(ApiBaseModel):
    in_features: Literal["int"] = "int"
    out_features: Literal["int"] = "int"

class TorchlinearArgs(ApiBaseModel):
  in_features: int
  out_features: int

class Torchlinear(ApiBaseModel):
    type: Literal['torch.nn.Linear'] = 'torch.nn.Linear'
    args_template: TorchlinearArgsTemplate

    @classmethod
    def create(cls, args: TorchlinearArgs):
        lib_cls = get_class_from_path_string(cls.type)
        return lib_cls(**args)

class TorchlinearLayer(ApiBaseModel):
    type: Literal['torch.nn.Linear'] = 'torch.nn.Linear'
    args: TorchlinearArgs
    input_layer: Optional[bool] = True
    id: str
    forward: str

