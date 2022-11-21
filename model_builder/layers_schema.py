############################################################################
# This code was autogenerated. Make changes to app/features/model/templates
############################################################################

from typing import List, Literal, Optional, Union

from pydantic import BaseModel

from model_builder.utils import CamelCaseModel, get_class_from_path_string


def get_module_name(classpath: str) -> str:
    return ".".join(classpath.split(".")[:-1])


def is_func(obj):
    return str(type(obj)) == "<class 'function'>"


class ModelbuilderonehotConstructorArgsSummary(BaseModel):
    """
    Summarizes what types are the arguments necessary to instantiate model_builder.layers.OneHot

    Generated code
    """


class ModelbuilderonehotForwardArgsSummary(BaseModel):
    """
    Maps to references for the the arguments of model_builder.layers.OneHot.forward or model_builder.layers.OneHot.__call__

    References can be names of layers/featurizers defined in the same model config or reference an attribute
    of the same component

    Generated code
    """

    x1 = "typing.Union[list[str], list[int]]"


class ModelbuilderonehotForwardArgsReferences(BaseModel):

    x1: str


class ModelbuilderonehotSummary(CamelCaseModel):
    type: Literal["model_builder.layers.OneHot"] = "model_builder.layers.OneHot"
    constructor_args_summary = ModelbuilderonehotConstructorArgsSummary()

    forward_args_summary = ModelbuilderonehotForwardArgsSummary()


class ModelbuilderonehotLayerConfig(CamelCaseModel):
    """
    Layer configuration.

    "type" is a discriminator field, and each possible value for it the
    args type will be mapped to the arguments of the respective class

    Generated code
    """

    type: Literal["model_builder.layers.OneHot"] = "model_builder.layers.OneHot"
    name: str

    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
            return lib_cls
        return lib_cls()

    forward_args: ModelbuilderonehotForwardArgsReferences


class ModelbuilderglobalpoolingConstructorArgsSummary(BaseModel):
    """
    Summarizes what types are the arguments necessary to instantiate model_builder.layers.GlobalPooling

    Generated code
    """

    aggr = "<class 'str'>"


class ModelbuilderglobalpoolingForwardArgsSummary(BaseModel):
    """
    Maps to references for the the arguments of model_builder.layers.GlobalPooling.forward or model_builder.layers.GlobalPooling.__call__

    References can be names of layers/featurizers defined in the same model config or reference an attribute
    of the same component

    Generated code
    """

    x = "<class 'torch.Tensor'>"
    batch = "typing.Optional[torch.Tensor]?"
    size = "typing.Optional[int]?"


class ModelbuilderglobalpoolingForwardArgsReferences(BaseModel):

    x: str

    batch: Optional[str] = None
    size: Optional[str] = None


class ModelbuilderglobalpoolingSummary(CamelCaseModel):
    type: Literal[
        "model_builder.layers.GlobalPooling"
    ] = "model_builder.layers.GlobalPooling"
    constructor_args_summary = ModelbuilderglobalpoolingConstructorArgsSummary()

    forward_args_summary = ModelbuilderglobalpoolingForwardArgsSummary()


class ModelbuilderglobalpoolingConstructorArgs(BaseModel):
    """
    Maps to the arguments of model_builder.layers.GlobalPooling

    Generated code
    """

    aggr: str


class ModelbuilderglobalpoolingLayerConfig(CamelCaseModel):
    """
    Layer configuration.

    "type" is a discriminator field, and each possible value for it the
    args type will be mapped to the arguments of the respective class

    Generated code
    """

    type: Literal[
        "model_builder.layers.GlobalPooling"
    ] = "model_builder.layers.GlobalPooling"
    name: str

    constructor_args: ModelbuilderglobalpoolingConstructorArgs

    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
            return lib_cls
        return lib_cls(**self.constructor_args.dict())

    forward_args: ModelbuilderglobalpoolingForwardArgsReferences


class ModelbuilderconcatConstructorArgsSummary(BaseModel):
    """
    Summarizes what types are the arguments necessary to instantiate model_builder.layers.Concat

    Generated code
    """

    dim = "<class 'int'>?"


class ModelbuilderconcatForwardArgsSummary(BaseModel):
    """
    Maps to references for the the arguments of model_builder.layers.Concat.forward or model_builder.layers.Concat.__call__

    References can be names of layers/featurizers defined in the same model config or reference an attribute
    of the same component

    Generated code
    """

    xs = "typing.List[torch.Tensor]"


class ModelbuilderconcatForwardArgsReferences(BaseModel):

    xs: List[str]


class ModelbuilderconcatSummary(CamelCaseModel):
    type: Literal["model_builder.layers.Concat"] = "model_builder.layers.Concat"
    constructor_args_summary = ModelbuilderconcatConstructorArgsSummary()

    forward_args_summary = ModelbuilderconcatForwardArgsSummary()


class ModelbuilderconcatConstructorArgs(BaseModel):
    """
    Maps to the arguments of model_builder.layers.Concat

    Generated code
    """

    dim: Optional[int] = 0


class ModelbuilderconcatLayerConfig(CamelCaseModel):
    """
    Layer configuration.

    "type" is a discriminator field, and each possible value for it the
    args type will be mapped to the arguments of the respective class

    Generated code
    """

    type: Literal["model_builder.layers.Concat"] = "model_builder.layers.Concat"
    name: str

    constructor_args: ModelbuilderconcatConstructorArgs

    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
            return lib_cls
        return lib_cls(**self.constructor_args.dict())

    forward_args: ModelbuilderconcatForwardArgsReferences


class TorchlinearConstructorArgsSummary(BaseModel):
    """
    Summarizes what types are the arguments necessary to instantiate torch.nn.Linear

    Generated code
    """

    in_features = "<class 'int'>"
    out_features = "<class 'int'>"
    bias = "<class 'bool'>?"


class TorchlinearForwardArgsSummary(BaseModel):
    """
    Maps to references for the the arguments of torch.nn.Linear.forward or torch.nn.Linear.__call__

    References can be names of layers/featurizers defined in the same model config or reference an attribute
    of the same component

    Generated code
    """

    input = "<class 'torch.Tensor'>"


class TorchlinearForwardArgsReferences(BaseModel):

    input: str


class TorchlinearSummary(CamelCaseModel):
    type: Literal["torch.nn.Linear"] = "torch.nn.Linear"
    constructor_args_summary = TorchlinearConstructorArgsSummary()

    forward_args_summary = TorchlinearForwardArgsSummary()


class TorchlinearConstructorArgs(BaseModel):
    """
    Maps to the arguments of torch.nn.Linear

    Generated code
    """

    in_features: int
    out_features: int
    bias: Optional[bool] = True


class TorchlinearLayerConfig(CamelCaseModel):
    """
    Layer configuration.

    "type" is a discriminator field, and each possible value for it the
    args type will be mapped to the arguments of the respective class

    Generated code
    """

    type: Literal["torch.nn.Linear"] = "torch.nn.Linear"
    name: str

    constructor_args: TorchlinearConstructorArgs

    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
            return lib_cls
        return lib_cls(**self.constructor_args.dict())

    forward_args: TorchlinearForwardArgsReferences


class TorchsigmoidConstructorArgsSummary(BaseModel):
    """
    Summarizes what types are the arguments necessary to instantiate torch.nn.Sigmoid

    Generated code
    """


class TorchsigmoidForwardArgsSummary(BaseModel):
    """
    Maps to references for the the arguments of torch.nn.Sigmoid.forward or torch.nn.Sigmoid.__call__

    References can be names of layers/featurizers defined in the same model config or reference an attribute
    of the same component

    Generated code
    """

    input = "<class 'torch.Tensor'>"


class TorchsigmoidForwardArgsReferences(BaseModel):

    input: str


class TorchsigmoidSummary(CamelCaseModel):
    type: Literal["torch.nn.Sigmoid"] = "torch.nn.Sigmoid"
    constructor_args_summary = TorchsigmoidConstructorArgsSummary()

    forward_args_summary = TorchsigmoidForwardArgsSummary()


class TorchsigmoidLayerConfig(CamelCaseModel):
    """
    Layer configuration.

    "type" is a discriminator field, and each possible value for it the
    args type will be mapped to the arguments of the respective class

    Generated code
    """

    type: Literal["torch.nn.Sigmoid"] = "torch.nn.Sigmoid"
    name: str

    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
            return lib_cls
        return lib_cls()

    forward_args: TorchsigmoidForwardArgsReferences


class TorchreluConstructorArgsSummary(BaseModel):
    """
    Summarizes what types are the arguments necessary to instantiate torch.nn.ReLU

    Generated code
    """

    inplace = "<class 'bool'>?"


class TorchreluForwardArgsSummary(BaseModel):
    """
    Maps to references for the the arguments of torch.nn.ReLU.forward or torch.nn.ReLU.__call__

    References can be names of layers/featurizers defined in the same model config or reference an attribute
    of the same component

    Generated code
    """

    input = "<class 'torch.Tensor'>"


class TorchreluForwardArgsReferences(BaseModel):

    input: str


class TorchreluSummary(CamelCaseModel):
    type: Literal["torch.nn.ReLU"] = "torch.nn.ReLU"
    constructor_args_summary = TorchreluConstructorArgsSummary()

    forward_args_summary = TorchreluForwardArgsSummary()


class TorchreluConstructorArgs(BaseModel):
    """
    Maps to the arguments of torch.nn.ReLU

    Generated code
    """

    inplace: Optional[bool] = False


class TorchreluLayerConfig(CamelCaseModel):
    """
    Layer configuration.

    "type" is a discriminator field, and each possible value for it the
    args type will be mapped to the arguments of the respective class

    Generated code
    """

    type: Literal["torch.nn.ReLU"] = "torch.nn.ReLU"
    name: str

    constructor_args: TorchreluConstructorArgs

    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
            return lib_cls
        return lib_cls(**self.constructor_args.dict())

    forward_args: TorchreluForwardArgsReferences


class TorchgeometricgcnconvConstructorArgsSummary(BaseModel):
    """
    Summarizes what types are the arguments necessary to instantiate torch_geometric.nn.GCNConv

    Generated code
    """

    in_channels = "<class 'int'>"
    out_channels = "<class 'int'>"
    improved = "<class 'bool'>?"
    cached = "<class 'bool'>?"
    add_self_loops = "<class 'bool'>?"
    normalize = "<class 'bool'>?"
    bias = "<class 'bool'>?"


class TorchgeometricgcnconvForwardArgsSummary(BaseModel):
    """
    Maps to references for the the arguments of torch_geometric.nn.GCNConv.forward or torch_geometric.nn.GCNConv.__call__

    References can be names of layers/featurizers defined in the same model config or reference an attribute
    of the same component

    Generated code
    """

    x = "<class 'torch.Tensor'>"
    edge_index = "typing.Union[torch.Tensor, torch_sparse.tensor.SparseTensor]"
    edge_weight = "typing.Optional[torch.Tensor]?"


class TorchgeometricgcnconvForwardArgsReferences(BaseModel):

    x: str

    edge_index: str

    edge_weight: Optional[str] = None


class TorchgeometricgcnconvSummary(CamelCaseModel):
    type: Literal["torch_geometric.nn.GCNConv"] = "torch_geometric.nn.GCNConv"
    constructor_args_summary = TorchgeometricgcnconvConstructorArgsSummary()

    forward_args_summary = TorchgeometricgcnconvForwardArgsSummary()


class TorchgeometricgcnconvConstructorArgs(BaseModel):
    """
    Maps to the arguments of torch_geometric.nn.GCNConv

    Generated code
    """

    in_channels: int
    out_channels: int
    improved: Optional[bool] = False
    cached: Optional[bool] = False
    add_self_loops: Optional[bool] = True
    normalize: Optional[bool] = True
    bias: Optional[bool] = True


class TorchgeometricgcnconvLayerConfig(CamelCaseModel):
    """
    Layer configuration.

    "type" is a discriminator field, and each possible value for it the
    args type will be mapped to the arguments of the respective class

    Generated code
    """

    type: Literal["torch_geometric.nn.GCNConv"] = "torch_geometric.nn.GCNConv"
    name: str

    constructor_args: TorchgeometricgcnconvConstructorArgs

    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
            return lib_cls
        return lib_cls(**self.constructor_args.dict())

    forward_args: TorchgeometricgcnconvForwardArgsReferences


class ModelbuildermoleculefeaturizerConstructorArgsSummary(BaseModel):
    """
    Summarizes what types are the arguments necessary to instantiate model_builder.featurizers.MoleculeFeaturizer

    Generated code
    """

    allow_unknown = "<class 'bool'>"
    sym_bond_list = "<class 'bool'>"
    per_atom_fragmentation = "<class 'bool'>"


class ModelbuildermoleculefeaturizerForwardArgsSummary(BaseModel):
    """
    Maps to references for the the arguments of model_builder.featurizers.MoleculeFeaturizer.forward or model_builder.featurizers.MoleculeFeaturizer.__call__

    References can be names of layers/featurizers defined in the same model config or reference an attribute
    of the same component

    Generated code
    """

    mol = "typing.Union[rdkit.Chem.rdchem.Mol, str]"


class ModelbuildermoleculefeaturizerForwardArgsReferences(BaseModel):

    mol: str


class ModelbuildermoleculefeaturizerSummary(CamelCaseModel):
    type: Literal[
        "model_builder.featurizers.MoleculeFeaturizer"
    ] = "model_builder.featurizers.MoleculeFeaturizer"
    constructor_args_summary = ModelbuildermoleculefeaturizerConstructorArgsSummary()

    forward_args_summary = ModelbuildermoleculefeaturizerForwardArgsSummary()


class ModelbuildermoleculefeaturizerConstructorArgs(BaseModel):
    """
    Maps to the arguments of model_builder.featurizers.MoleculeFeaturizer

    Generated code
    """

    allow_unknown: bool
    sym_bond_list: bool
    per_atom_fragmentation: bool


class ModelbuildermoleculefeaturizerLayerConfig(CamelCaseModel):
    """
    Layer configuration.

    "type" is a discriminator field, and each possible value for it the
    args type will be mapped to the arguments of the respective class

    Generated code
    """

    type: Literal[
        "model_builder.featurizers.MoleculeFeaturizer"
    ] = "model_builder.featurizers.MoleculeFeaturizer"
    name: str

    constructor_args: ModelbuildermoleculefeaturizerConstructorArgs

    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
            return lib_cls
        return lib_cls(**self.constructor_args.dict())

    forward_args: ModelbuildermoleculefeaturizerForwardArgsReferences


LayersType = Union[
    ModelbuilderonehotLayerConfig,
    ModelbuilderglobalpoolingLayerConfig,
    ModelbuilderconcatLayerConfig,
    TorchlinearLayerConfig,
    TorchsigmoidLayerConfig,
    TorchreluLayerConfig,
    TorchgeometricgcnconvLayerConfig,
]

FeaturizersType = Union[
    ModelbuildermoleculefeaturizerLayerConfig,
]

LayersArgsType = Union[
    ModelbuilderonehotSummary,
    ModelbuilderglobalpoolingSummary,
    ModelbuilderconcatSummary,
    TorchlinearSummary,
    TorchsigmoidSummary,
    TorchreluSummary,
    TorchgeometricgcnconvSummary,
]

FeaturizersArgsType = Union[
    ModelbuildermoleculefeaturizerSummary,
]
