# flake8: noqa
############################################################################
# This code was autogenerated. Make changes to app/features/model/templates
############################################################################

from typing import Callable, List, Literal, Optional, Union

import torch
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
        args = self.constructor_args.dict()
        newargs = {}
        for key, value in args.items():
            if value is not None:
                newargs[key] = value
        return lib_cls(**newargs)

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

    dim: Optional[int] = None


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
        args = self.constructor_args.dict()
        newargs = {}
        for key, value in args.items():
            if value is not None:
                newargs[key] = value
        return lib_cls(**newargs)

    forward_args: ModelbuilderconcatForwardArgsReferences


class ModelbuilderaddpoolingConstructorArgsSummary(BaseModel):
    """
    Summarizes what types are the arguments necessary to instantiate model_builder.layers.AddPooling

    Generated code
    """

    dim = "typing.Optional[int]?"


class ModelbuilderaddpoolingForwardArgsSummary(BaseModel):
    """
    Maps to references for the the arguments of model_builder.layers.AddPooling.forward or model_builder.layers.AddPooling.__call__

    References can be names of layers/featurizers defined in the same model config or reference an attribute
    of the same component

    Generated code
    """

    x = "<class 'torch.Tensor'>"


class ModelbuilderaddpoolingForwardArgsReferences(BaseModel):

    x: str


class ModelbuilderaddpoolingSummary(CamelCaseModel):
    type: Literal["model_builder.layers.AddPooling"] = "model_builder.layers.AddPooling"
    constructor_args_summary = ModelbuilderaddpoolingConstructorArgsSummary()

    forward_args_summary = ModelbuilderaddpoolingForwardArgsSummary()


class ModelbuilderaddpoolingConstructorArgs(BaseModel):
    """
    Maps to the arguments of model_builder.layers.AddPooling

    Generated code
    """

    dim: Optional[Optional[int]] = None


class ModelbuilderaddpoolingLayerConfig(CamelCaseModel):
    """
    Layer configuration.

    "type" is a discriminator field, and each possible value for it the
    args type will be mapped to the arguments of the respective class

    Generated code
    """

    type: Literal["model_builder.layers.AddPooling"] = "model_builder.layers.AddPooling"
    name: str

    constructor_args: ModelbuilderaddpoolingConstructorArgs

    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
            return lib_cls
        args = self.constructor_args.dict()
        newargs = {}
        for key, value in args.items():
            if value is not None:
                newargs[key] = value
        return lib_cls(**newargs)

    forward_args: ModelbuilderaddpoolingForwardArgsReferences


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
    bias: Optional[bool] = None


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
        args = self.constructor_args.dict()
        newargs = {}
        for key, value in args.items():
            if value is not None:
                newargs[key] = value
        return lib_cls(**newargs)

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

    inplace: Optional[bool] = None


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
        args = self.constructor_args.dict()
        newargs = {}
        for key, value in args.items():
            if value is not None:
                newargs[key] = value
        return lib_cls(**newargs)

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
    improved: Optional[bool] = None
    cached: Optional[bool] = None
    add_self_loops: Optional[bool] = None
    normalize: Optional[bool] = None
    bias: Optional[bool] = None


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
        args = self.constructor_args.dict()
        newargs = {}
        for key, value in args.items():
            if value is not None:
                newargs[key] = value
        return lib_cls(**newargs)

    forward_args: TorchgeometricgcnconvForwardArgsReferences


class TorchembeddingConstructorArgsSummary(BaseModel):
    """
    Summarizes what types are the arguments necessary to instantiate torch.nn.Embedding

    Generated code
    """

    num_embeddings = "<class 'int'>"
    embedding_dim = "<class 'int'>"
    padding_idx = "typing.Optional[int]?"
    max_norm = "typing.Optional[float]?"
    norm_type = "<class 'float'>?"
    scale_grad_by_freq = "<class 'bool'>?"
    sparse = "<class 'bool'>?"
    _weight = "typing.Optional[torch.Tensor]?"


class TorchembeddingForwardArgsSummary(BaseModel):
    """
    Maps to references for the the arguments of torch.nn.Embedding.forward or torch.nn.Embedding.__call__

    References can be names of layers/featurizers defined in the same model config or reference an attribute
    of the same component

    Generated code
    """

    input = "<class 'torch.Tensor'>"


class TorchembeddingForwardArgsReferences(BaseModel):

    input: str


class TorchembeddingSummary(CamelCaseModel):
    type: Literal["torch.nn.Embedding"] = "torch.nn.Embedding"
    constructor_args_summary = TorchembeddingConstructorArgsSummary()

    forward_args_summary = TorchembeddingForwardArgsSummary()


class TorchembeddingConstructorArgs(BaseModel):
    """
    Maps to the arguments of torch.nn.Embedding

    Generated code
    """

    num_embeddings: int
    embedding_dim: int
    padding_idx: Optional[Optional[int]] = None
    max_norm: Optional[Optional[float]] = None
    norm_type: Optional[float] = None
    scale_grad_by_freq: Optional[bool] = None
    sparse: Optional[bool] = None
    _weight: Optional[Optional[torch.Tensor]] = None


class TorchembeddingLayerConfig(CamelCaseModel):
    """
    Layer configuration.

    "type" is a discriminator field, and each possible value for it the
    args type will be mapped to the arguments of the respective class

    Generated code
    """

    type: Literal["torch.nn.Embedding"] = "torch.nn.Embedding"
    name: str

    constructor_args: TorchembeddingConstructorArgs

    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
            return lib_cls
        args = self.constructor_args.dict()
        newargs = {}
        for key, value in args.items():
            if value is not None:
                newargs[key] = value
        return lib_cls(**newargs)

    forward_args: TorchembeddingForwardArgsReferences


class TorchtransformerencoderlayerConstructorArgsSummary(BaseModel):
    """
    Summarizes what types are the arguments necessary to instantiate torch.nn.TransformerEncoderLayer

    Generated code
    """

    d_model = "<class 'int'>"
    nhead = "<class 'int'>"
    dim_feedforward = "<class 'int'>?"
    dropout = "<class 'float'>?"
    activation = "typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]?"
    layer_norm_eps = "<class 'float'>?"
    batch_first = "<class 'bool'>?"
    norm_first = "<class 'bool'>?"


class TorchtransformerencoderlayerForwardArgsSummary(BaseModel):
    """
    Maps to references for the the arguments of torch.nn.TransformerEncoderLayer.forward or torch.nn.TransformerEncoderLayer.__call__

    References can be names of layers/featurizers defined in the same model config or reference an attribute
    of the same component

    Generated code
    """

    src = "<class 'torch.Tensor'>"
    src_mask = "typing.Optional[torch.Tensor]?"
    src_key_padding_mask = "typing.Optional[torch.Tensor]?"


class TorchtransformerencoderlayerForwardArgsReferences(BaseModel):

    src: str

    src_mask: Optional[str] = None
    src_key_padding_mask: Optional[str] = None


class TorchtransformerencoderlayerSummary(CamelCaseModel):
    type: Literal[
        "torch.nn.TransformerEncoderLayer"
    ] = "torch.nn.TransformerEncoderLayer"
    constructor_args_summary = TorchtransformerencoderlayerConstructorArgsSummary()

    forward_args_summary = TorchtransformerencoderlayerForwardArgsSummary()


class TorchtransformerencoderlayerConstructorArgs(BaseModel):
    """
    Maps to the arguments of torch.nn.TransformerEncoderLayer

    Generated code
    """

    d_model: int
    nhead: int
    dim_feedforward: Optional[int] = None
    dropout: Optional[float] = None
    activation: Optional[Union[str, Callable[[torch.Tensor], torch.Tensor]]] = None
    layer_norm_eps: Optional[float] = None
    batch_first: Optional[bool] = None
    norm_first: Optional[bool] = None


class TorchtransformerencoderlayerLayerConfig(CamelCaseModel):
    """
    Layer configuration.

    "type" is a discriminator field, and each possible value for it the
    args type will be mapped to the arguments of the respective class

    Generated code
    """

    type: Literal[
        "torch.nn.TransformerEncoderLayer"
    ] = "torch.nn.TransformerEncoderLayer"
    name: str

    constructor_args: TorchtransformerencoderlayerConstructorArgs

    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
            return lib_cls
        args = self.constructor_args.dict()
        newargs = {}
        for key, value in args.items():
            if value is not None:
                newargs[key] = value
        return lib_cls(**newargs)

    forward_args: TorchtransformerencoderlayerForwardArgsReferences


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
        args = self.constructor_args.dict()
        newargs = {}
        for key, value in args.items():
            if value is not None:
                newargs[key] = value
        return lib_cls(**newargs)

    forward_args: ModelbuildermoleculefeaturizerForwardArgsReferences


class ModelbuilderintegerfeaturizerConstructorArgsSummary(BaseModel):
    """
    Summarizes what types are the arguments necessary to instantiate model_builder.featurizers.IntegerFeaturizer

    Generated code
    """


class ModelbuilderintegerfeaturizerForwardArgsSummary(BaseModel):
    """
    Maps to references for the the arguments of model_builder.featurizers.IntegerFeaturizer.forward or model_builder.featurizers.IntegerFeaturizer.__call__

    References can be names of layers/featurizers defined in the same model config or reference an attribute
    of the same component

    Generated code
    """

    input_ = "<class 'str'>"


class ModelbuilderintegerfeaturizerForwardArgsReferences(BaseModel):

    input_: str


class ModelbuilderintegerfeaturizerSummary(CamelCaseModel):
    type: Literal[
        "model_builder.featurizers.IntegerFeaturizer"
    ] = "model_builder.featurizers.IntegerFeaturizer"
    constructor_args_summary = ModelbuilderintegerfeaturizerConstructorArgsSummary()

    forward_args_summary = ModelbuilderintegerfeaturizerForwardArgsSummary()


class ModelbuilderintegerfeaturizerLayerConfig(CamelCaseModel):
    """
    Layer configuration.

    "type" is a discriminator field, and each possible value for it the
    args type will be mapped to the arguments of the respective class

    Generated code
    """

    type: Literal[
        "model_builder.featurizers.IntegerFeaturizer"
    ] = "model_builder.featurizers.IntegerFeaturizer"
    name: str

    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
            return lib_cls
        return lib_cls()

    forward_args: ModelbuilderintegerfeaturizerForwardArgsReferences


class ModelbuilderdnasequencefeaturizerConstructorArgsSummary(BaseModel):
    """
    Summarizes what types are the arguments necessary to instantiate model_builder.featurizers.DNASequenceFeaturizer

    Generated code
    """


class ModelbuilderdnasequencefeaturizerForwardArgsSummary(BaseModel):
    """
    Maps to references for the the arguments of model_builder.featurizers.DNASequenceFeaturizer.forward or model_builder.featurizers.DNASequenceFeaturizer.__call__

    References can be names of layers/featurizers defined in the same model config or reference an attribute
    of the same component

    Generated code
    """

    input_ = "<class 'str'>"


class ModelbuilderdnasequencefeaturizerForwardArgsReferences(BaseModel):

    input_: str


class ModelbuilderdnasequencefeaturizerSummary(CamelCaseModel):
    type: Literal[
        "model_builder.featurizers.DNASequenceFeaturizer"
    ] = "model_builder.featurizers.DNASequenceFeaturizer"
    constructor_args_summary = ModelbuilderdnasequencefeaturizerConstructorArgsSummary()

    forward_args_summary = ModelbuilderdnasequencefeaturizerForwardArgsSummary()


class ModelbuilderdnasequencefeaturizerLayerConfig(CamelCaseModel):
    """
    Layer configuration.

    "type" is a discriminator field, and each possible value for it the
    args type will be mapped to the arguments of the respective class

    Generated code
    """

    type: Literal[
        "model_builder.featurizers.DNASequenceFeaturizer"
    ] = "model_builder.featurizers.DNASequenceFeaturizer"
    name: str

    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
            return lib_cls
        return lib_cls()

    forward_args: ModelbuilderdnasequencefeaturizerForwardArgsReferences


class ModelbuilderrnasequencefeaturizerConstructorArgsSummary(BaseModel):
    """
    Summarizes what types are the arguments necessary to instantiate model_builder.featurizers.RNASequenceFeaturizer

    Generated code
    """


class ModelbuilderrnasequencefeaturizerForwardArgsSummary(BaseModel):
    """
    Maps to references for the the arguments of model_builder.featurizers.RNASequenceFeaturizer.forward or model_builder.featurizers.RNASequenceFeaturizer.__call__

    References can be names of layers/featurizers defined in the same model config or reference an attribute
    of the same component

    Generated code
    """

    input_ = "<class 'str'>"


class ModelbuilderrnasequencefeaturizerForwardArgsReferences(BaseModel):

    input_: str


class ModelbuilderrnasequencefeaturizerSummary(CamelCaseModel):
    type: Literal[
        "model_builder.featurizers.RNASequenceFeaturizer"
    ] = "model_builder.featurizers.RNASequenceFeaturizer"
    constructor_args_summary = ModelbuilderrnasequencefeaturizerConstructorArgsSummary()

    forward_args_summary = ModelbuilderrnasequencefeaturizerForwardArgsSummary()


class ModelbuilderrnasequencefeaturizerLayerConfig(CamelCaseModel):
    """
    Layer configuration.

    "type" is a discriminator field, and each possible value for it the
    args type will be mapped to the arguments of the respective class

    Generated code
    """

    type: Literal[
        "model_builder.featurizers.RNASequenceFeaturizer"
    ] = "model_builder.featurizers.RNASequenceFeaturizer"
    name: str

    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
            return lib_cls
        return lib_cls()

    forward_args: ModelbuilderrnasequencefeaturizerForwardArgsReferences


class ModelbuilderproteinsequencefeaturizerConstructorArgsSummary(BaseModel):
    """
    Summarizes what types are the arguments necessary to instantiate model_builder.featurizers.ProteinSequenceFeaturizer

    Generated code
    """


class ModelbuilderproteinsequencefeaturizerForwardArgsSummary(BaseModel):
    """
    Maps to references for the the arguments of model_builder.featurizers.ProteinSequenceFeaturizer.forward or model_builder.featurizers.ProteinSequenceFeaturizer.__call__

    References can be names of layers/featurizers defined in the same model config or reference an attribute
    of the same component

    Generated code
    """

    input_ = "<class 'str'>"


class ModelbuilderproteinsequencefeaturizerForwardArgsReferences(BaseModel):

    input_: str


class ModelbuilderproteinsequencefeaturizerSummary(CamelCaseModel):
    type: Literal[
        "model_builder.featurizers.ProteinSequenceFeaturizer"
    ] = "model_builder.featurizers.ProteinSequenceFeaturizer"
    constructor_args_summary = (
        ModelbuilderproteinsequencefeaturizerConstructorArgsSummary()
    )

    forward_args_summary = ModelbuilderproteinsequencefeaturizerForwardArgsSummary()


class ModelbuilderproteinsequencefeaturizerLayerConfig(CamelCaseModel):
    """
    Layer configuration.

    "type" is a discriminator field, and each possible value for it the
    args type will be mapped to the arguments of the respective class

    Generated code
    """

    type: Literal[
        "model_builder.featurizers.ProteinSequenceFeaturizer"
    ] = "model_builder.featurizers.ProteinSequenceFeaturizer"
    name: str

    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
            return lib_cls
        return lib_cls()

    forward_args: ModelbuilderproteinsequencefeaturizerForwardArgsReferences


LayersType = Union[
    ModelbuilderonehotLayerConfig,
    ModelbuilderglobalpoolingLayerConfig,
    ModelbuilderconcatLayerConfig,
    ModelbuilderaddpoolingLayerConfig,
    TorchlinearLayerConfig,
    TorchsigmoidLayerConfig,
    TorchreluLayerConfig,
    TorchgeometricgcnconvLayerConfig,
    TorchembeddingLayerConfig,
    TorchtransformerencoderlayerLayerConfig,
]

FeaturizersType = Union[
    ModelbuildermoleculefeaturizerLayerConfig,
    ModelbuilderintegerfeaturizerLayerConfig,
    ModelbuilderdnasequencefeaturizerLayerConfig,
    ModelbuilderrnasequencefeaturizerLayerConfig,
    ModelbuilderproteinsequencefeaturizerLayerConfig,
]

LayersArgsType = Union[
    ModelbuilderonehotSummary,
    ModelbuilderglobalpoolingSummary,
    ModelbuilderconcatSummary,
    ModelbuilderaddpoolingSummary,
    TorchlinearSummary,
    TorchsigmoidSummary,
    TorchreluSummary,
    TorchgeometricgcnconvSummary,
    TorchembeddingSummary,
    TorchtransformerencoderlayerSummary,
]

FeaturizersArgsType = Union[
    ModelbuildermoleculefeaturizerSummary,
    ModelbuilderintegerfeaturizerSummary,
    ModelbuilderdnasequencefeaturizerSummary,
    ModelbuilderrnasequencefeaturizerSummary,
    ModelbuilderproteinsequencefeaturizerSummary,
]
