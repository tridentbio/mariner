###########################################################################
# This is some auto generated code. Your changes may not take effect.     #
# Consider making the changes im `app/features/model/templates`           #
###########################################################################

from typing import Literal, Optional, Union

from app.features.model.utils import get_class_from_path_string
from app.schemas.api import ApiBaseModel


def get_module_name(classpath: str) -> str:
    return ".".join(classpath.split(".")[:-1])


class TorchlinearArgsTemplate(ApiBaseModel):
    in_features: Literal["int"] = "int"
    out_features: Literal["int"] = "int"


class TorchlinearArgs(ApiBaseModel):
    in_features: int
    out_features: int


class TorchlinearLayerConfig(ApiBaseModel):
    type: Literal["torch.nn.Linear"] = "torch.nn.Linear"

    args: TorchlinearArgs

    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        return lib_cls(**self.args.dict())

    input_layer: Optional[bool] = True
    name: str
    forward: Optional[str]


class TorchflattenLayerConfig(ApiBaseModel):
    type: Literal["torch.nn.Flatten"] = "torch.nn.Flatten"

    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        return lib_cls()

    input_layer: Optional[bool] = True
    name: str
    forward: Optional[str]


class TorchgeometricginconvLayerConfig(ApiBaseModel):
    type: Literal["torch_geometric.nn.GINConv"] = "torch_geometric.nn.GINConv"

    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        return lib_cls()

    input_layer: Optional[bool] = True
    name: str
    forward: Optional[str]


class TorchgeometricgcnconvArgsTemplate(ApiBaseModel):
    in_channels: Literal["int"] = "int"
    out_channels: Literal["int"] = "int"


class TorchgeometricgcnconvArgs(ApiBaseModel):
    in_channels: int
    out_channels: int


class TorchgeometricgcnconvLayerConfig(ApiBaseModel):
    type: Literal["torch_geometric.nn.GCNConv"] = "torch_geometric.nn.GCNConv"

    args: TorchgeometricgcnconvArgs

    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        return lib_cls(**self.args.dict())

    input_layer: Optional[bool] = True
    name: str
    forward: Optional[str]


class TorchgeometricglobaladdpoolLayerConfig(ApiBaseModel):
    type: Literal[
        "torch_geometric.nn.global_add_pool"
    ] = "torch_geometric.nn.global_add_pool"

    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        return lib_cls()

    input_layer: Optional[bool] = True
    name: str
    forward: Optional[str]


LayersType = Union[
    TorchlinearLayerConfig,
    TorchflattenLayerConfig,
    TorchgeometricginconvLayerConfig,
    TorchgeometricgcnconvLayerConfig,
    TorchgeometricglobaladdpoolLayerConfig,
]
