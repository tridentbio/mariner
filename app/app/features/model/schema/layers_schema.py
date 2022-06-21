###########################################################################
# This is some auto generated code. Your changes may not have effect.     #
# Consider making the changes im `app/features/model/templates`           #
###########################################################################

from typing import Optional, Literal, Union, List
from app.features.model.utils import get_class_from_path_string
from app.schemas.api import ApiBaseModel



def get_module_name(classpath: str) -> str:
    return '.'.join(classpath.split('.')[:-1])

def is_func(obj):
  return str(type(obj)) == "<class 'function'>"


class BaseLayerConfig(ApiBaseModel):
    name: str
    input: Union[str, List[str]]
    def create(self):
      pass



class AppglobalpoolingArgsTemplate(ApiBaseModel):
    type: Literal['app.features.model.layers.GlobalPooling'] = 'app.features.model.layers.GlobalPooling'
      aggr: Literal["string"] = "string"

class AppglobalpoolingArgs(ApiBaseModel):
    aggr: str



class AppglobalpoolingLayerConfig(BaseLayerConfig):
    type: Literal['app.features.model.layers.GlobalPooling'] = 'app.features.model.layers.GlobalPooling'
    
    args: AppglobalpoolingArgs
    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
          return lib_cls
        return lib_cls(**self.args.dict())
    





class AppconcatLayerConfig(BaseLayerConfig):
    type: Literal['app.features.model.layers.Concat'] = 'app.features.model.layers.Concat'
    
    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
          return lib_cls
        return lib_cls()
    





class TorchlinearArgsTemplate(ApiBaseModel):
    type: Literal['torch.nn.Linear'] = 'torch.nn.Linear'
      in_features: Literal["int"] = "int"
      out_features: Literal["int"] = "int"

class TorchlinearArgs(ApiBaseModel):
    in_features: int
    out_features: int



class TorchlinearLayerConfig(BaseLayerConfig):
    type: Literal['torch.nn.Linear'] = 'torch.nn.Linear'
    
    args: TorchlinearArgs
    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
          return lib_cls
        return lib_cls(**self.args.dict())
    





class TorchsigmoidLayerConfig(BaseLayerConfig):
    type: Literal['torch.nn.Sigmoid'] = 'torch.nn.Sigmoid'
    
    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
          return lib_cls
        return lib_cls()
    





class TorchreluLayerConfig(BaseLayerConfig):
    type: Literal['torch.nn.ReLU'] = 'torch.nn.ReLU'
    
    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
          return lib_cls
        return lib_cls()
    





class TorchgeometricgcnconvArgsTemplate(ApiBaseModel):
    type: Literal['torch_geometric.nn.GCNConv'] = 'torch_geometric.nn.GCNConv'
      in_channels: Literal["int"] = "int"
      out_channels: Literal["int"] = "int"

class TorchgeometricgcnconvArgs(ApiBaseModel):
    in_channels: int
    out_channels: int



class TorchgeometricgcnconvLayerConfig(BaseLayerConfig):
    type: Literal['torch_geometric.nn.GCNConv'] = 'torch_geometric.nn.GCNConv'
    
    args: TorchgeometricgcnconvArgs
    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
          return lib_cls
        return lib_cls(**self.args.dict())
    





class AppmoleculefeaturizerLayerConfig(BaseLayerConfig):
    type: Literal['app.features.model.featurizers.MoleculeFeaturizer'] = 'app.features.model.featurizers.MoleculeFeaturizer'
    
    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
          return lib_cls
        return lib_cls()
    



LayersType = Union[
    AppglobalpoolingLayerConfig,
    AppconcatLayerConfig,
    TorchlinearLayerConfig,
    TorchsigmoidLayerConfig,
    TorchreluLayerConfig,
    TorchgeometricgcnconvLayerConfig,
    
]

FeaturizersType = Union[
    AppmoleculefeaturizerLayerConfig,
    
]

