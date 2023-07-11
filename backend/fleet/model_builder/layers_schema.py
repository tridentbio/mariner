# flake8: noqa
############################################################################
# This code was autogenerated. Make changes to app/features/model/templates
############################################################################

from typing import Any, Callable, List, Literal, Optional, Union

import torch
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from fleet.model_builder.utils import CamelCaseModel, get_class_from_path_string
from fleet.options import options_manager


def get_module_name(classpath: str) -> str:
    return ".".join(classpath.split(".")[:-1])


def is_func(obj):
    return str(type(obj)) == "<class 'function'>"


class FleetonehotConstructorArgsSummary(BaseModel):
    """
    Summarizes what types are the arguments necessary to instantiate fleet.model_builder.layers.OneHot

    Generated code
    """


class FleetonehotForwardArgsSummary(BaseModel):
    """
    Maps to references for the the arguments of fleet.model_builder.layers.OneHot.forward or fleet.model_builder.layers.OneHot.__call__

    References can be names of layers/featurizers defined in the same model config or reference an attribute
    of the same component

    Generated code
    """

    x1 = "typing.Union[list[str], list[int]]"


class FleetonehotForwardArgsReferences(BaseModel):

    x1: str


class FleetonehotSummary(CamelCaseModel):
    type: Literal[
        "fleet.model_builder.layers.OneHot"
    ] = "fleet.model_builder.layers.OneHot"
    constructor_args_summary = FleetonehotConstructorArgsSummary()

    forward_args_summary = FleetonehotForwardArgsSummary()


@options_manager.config_layer(summary_cls=FleetonehotSummary)
class FleetonehotLayerConfig(CamelCaseModel):
    """
    Layer configuration.

    "type" is a discriminator field, and each possible value for it the
    args type will be mapped to the arguments of the respective class

    Generated code
    """

    type: Literal[
        "fleet.model_builder.layers.OneHot"
    ] = "fleet.model_builder.layers.OneHot"
    name: str

    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
            return lib_cls
        return lib_cls()

    forward_args: FleetonehotForwardArgsReferences


class FleetglobalpoolingConstructorArgsSummary(BaseModel):
    """
    Summarizes what types are the arguments necessary to instantiate fleet.model_builder.layers.GlobalPooling

    Generated code
    """

    aggr = "<class 'str'>"


class FleetglobalpoolingForwardArgsSummary(BaseModel):
    """
    Maps to references for the the arguments of fleet.model_builder.layers.GlobalPooling.forward or fleet.model_builder.layers.GlobalPooling.__call__

    References can be names of layers/featurizers defined in the same model config or reference an attribute
    of the same component

    Generated code
    """

    x = "<class 'torch.Tensor'>"
    batch = "typing.Optional[torch.Tensor]?"
    size = "typing.Optional[int]?"


class FleetglobalpoolingForwardArgsReferences(BaseModel):

    x: str

    batch: Optional[str] = None
    size: Optional[str] = None


class FleetglobalpoolingSummary(CamelCaseModel):
    type: Literal[
        "fleet.model_builder.layers.GlobalPooling"
    ] = "fleet.model_builder.layers.GlobalPooling"
    constructor_args_summary = FleetglobalpoolingConstructorArgsSummary()

    forward_args_summary = FleetglobalpoolingForwardArgsSummary()


class FleetglobalpoolingConstructorArgs(BaseModel):
    """
    Maps to the arguments of fleet.model_builder.layers.GlobalPooling

    Generated code
    """

    aggr: str


@options_manager.config_layer(summary_cls=FleetglobalpoolingSummary)
class FleetglobalpoolingLayerConfig(CamelCaseModel):
    """
    Layer configuration.

    "type" is a discriminator field, and each possible value for it the
    args type will be mapped to the arguments of the respective class

    Generated code
    """

    type: Literal[
        "fleet.model_builder.layers.GlobalPooling"
    ] = "fleet.model_builder.layers.GlobalPooling"
    name: str

    constructor_args: FleetglobalpoolingConstructorArgs

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

    forward_args: FleetglobalpoolingForwardArgsReferences


class FleetconcatConstructorArgsSummary(BaseModel):
    """
    Summarizes what types are the arguments necessary to instantiate fleet.model_builder.layers.Concat

    Generated code
    """

    dim = "<class 'int'>?"


class FleetconcatForwardArgsSummary(BaseModel):
    """
    Maps to references for the the arguments of fleet.model_builder.layers.Concat.forward or fleet.model_builder.layers.Concat.__call__

    References can be names of layers/featurizers defined in the same model config or reference an attribute
    of the same component

    Generated code
    """

    xs = "typing.List[torch.Tensor]"


class FleetconcatForwardArgsReferences(BaseModel):

    xs: List[str]


class FleetconcatSummary(CamelCaseModel):
    type: Literal[
        "fleet.model_builder.layers.Concat"
    ] = "fleet.model_builder.layers.Concat"
    constructor_args_summary = FleetconcatConstructorArgsSummary()

    forward_args_summary = FleetconcatForwardArgsSummary()


class FleetconcatConstructorArgs(BaseModel):
    """
    Maps to the arguments of fleet.model_builder.layers.Concat

    Generated code
    """

    dim: Optional[int] = 0


@options_manager.config_layer(summary_cls=FleetconcatSummary)
class FleetconcatLayerConfig(CamelCaseModel):
    """
    Layer configuration.

    "type" is a discriminator field, and each possible value for it the
    args type will be mapped to the arguments of the respective class

    Generated code
    """

    type: Literal[
        "fleet.model_builder.layers.Concat"
    ] = "fleet.model_builder.layers.Concat"
    name: str

    constructor_args: FleetconcatConstructorArgs

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

    forward_args: FleetconcatForwardArgsReferences


class FleetaddpoolingConstructorArgsSummary(BaseModel):
    """
    Summarizes what types are the arguments necessary to instantiate fleet.model_builder.layers.AddPooling

    Generated code
    """

    dim = "typing.Optional[int]?"


class FleetaddpoolingForwardArgsSummary(BaseModel):
    """
    Maps to references for the the arguments of fleet.model_builder.layers.AddPooling.forward or fleet.model_builder.layers.AddPooling.__call__

    References can be names of layers/featurizers defined in the same model config or reference an attribute
    of the same component

    Generated code
    """

    x = "<class 'torch.Tensor'>"


class FleetaddpoolingForwardArgsReferences(BaseModel):

    x: str


class FleetaddpoolingSummary(CamelCaseModel):
    type: Literal[
        "fleet.model_builder.layers.AddPooling"
    ] = "fleet.model_builder.layers.AddPooling"
    constructor_args_summary = FleetaddpoolingConstructorArgsSummary()

    forward_args_summary = FleetaddpoolingForwardArgsSummary()


class FleetaddpoolingConstructorArgs(BaseModel):
    """
    Maps to the arguments of fleet.model_builder.layers.AddPooling

    Generated code
    """

    dim: Optional[Optional[int]] = None


@options_manager.config_layer(summary_cls=FleetaddpoolingSummary)
class FleetaddpoolingLayerConfig(CamelCaseModel):
    """
    Layer configuration.

    "type" is a discriminator field, and each possible value for it the
    args type will be mapped to the arguments of the respective class

    Generated code
    """

    type: Literal[
        "fleet.model_builder.layers.AddPooling"
    ] = "fleet.model_builder.layers.AddPooling"
    name: str

    constructor_args: FleetaddpoolingConstructorArgs

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

    forward_args: FleetaddpoolingForwardArgsReferences


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


@options_manager.config_layer(summary_cls=TorchlinearSummary)
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


@options_manager.config_layer(summary_cls=TorchsigmoidSummary)
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


@options_manager.config_layer(summary_cls=TorchreluSummary)
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
    improved: Optional[bool] = False
    cached: Optional[bool] = False
    add_self_loops: Optional[bool] = True
    normalize: Optional[bool] = True
    bias: Optional[bool] = True


@options_manager.config_layer(summary_cls=TorchgeometricgcnconvSummary)
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
    _freeze = "<class 'bool'>?"


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
    norm_type: Optional[float] = 2.0
    scale_grad_by_freq: Optional[bool] = False
    sparse: Optional[bool] = False
    _weight: Optional[Optional[torch.Tensor]] = None
    _freeze: Optional[bool] = False


@options_manager.config_layer(summary_cls=TorchembeddingSummary)
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
    activation = (
        "typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]?"
    )
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
    is_causal = "<class 'bool'>?"


class TorchtransformerencoderlayerForwardArgsReferences(BaseModel):

    src: str

    src_mask: Optional[str] = None
    src_key_padding_mask: Optional[str] = None
    is_causal: Optional[str] = None


class TorchtransformerencoderlayerSummary(CamelCaseModel):
    type: Literal[
        "torch.nn.TransformerEncoderLayer"
    ] = "torch.nn.TransformerEncoderLayer"
    constructor_args_summary = (
        TorchtransformerencoderlayerConstructorArgsSummary()
    )

    forward_args_summary = TorchtransformerencoderlayerForwardArgsSummary()


class TorchtransformerencoderlayerConstructorArgs(BaseModel):
    """
    Maps to the arguments of torch.nn.TransformerEncoderLayer

    Generated code
    """

    d_model: int
    nhead: int
    dim_feedforward: Optional[int] = 2048
    dropout: Optional[float] = 0.1
    activation: Optional[
        Union[str, Callable[[torch.Tensor], torch.Tensor]]
    ] = None
    layer_norm_eps: Optional[float] = 1e-05
    batch_first: Optional[bool] = False
    norm_first: Optional[bool] = False


@options_manager.config_layer(summary_cls=TorchtransformerencoderlayerSummary)
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


class FleetmoleculefeaturizerConstructorArgsSummary(BaseModel):
    """
    Summarizes what types are the arguments necessary to instantiate fleet.model_builder.featurizers.MoleculeFeaturizer

    Generated code
    """

    allow_unknown = "<class 'bool'>"
    sym_bond_list = "<class 'bool'>"
    per_atom_fragmentation = "<class 'bool'>"


class FleetmoleculefeaturizerForwardArgsSummary(BaseModel):
    """
    Maps to references for the the arguments of fleet.model_builder.featurizers.MoleculeFeaturizer.forward or fleet.model_builder.featurizers.MoleculeFeaturizer.__call__

    References can be names of layers/featurizers defined in the same model config or reference an attribute
    of the same component

    Generated code
    """

    mol = "typing.Union[rdkit.Chem.rdchem.Mol, str, list]"


class FleetmoleculefeaturizerForwardArgsReferences(BaseModel):

    mol: str


class FleetmoleculefeaturizerSummary(CamelCaseModel):
    type: Literal[
        "fleet.model_builder.featurizers.MoleculeFeaturizer"
    ] = "fleet.model_builder.featurizers.MoleculeFeaturizer"
    constructor_args_summary = FleetmoleculefeaturizerConstructorArgsSummary()

    forward_args_summary = FleetmoleculefeaturizerForwardArgsSummary()


class FleetmoleculefeaturizerConstructorArgs(BaseModel):
    """
    Maps to the arguments of fleet.model_builder.featurizers.MoleculeFeaturizer

    Generated code
    """

    allow_unknown: bool
    sym_bond_list: bool
    per_atom_fragmentation: bool


@options_manager.config_featurizer(summary_cls=FleetmoleculefeaturizerSummary)
class FleetmoleculefeaturizerLayerConfig(CamelCaseModel):
    """
    Layer configuration.

    "type" is a discriminator field, and each possible value for it the
    args type will be mapped to the arguments of the respective class

    Generated code
    """

    type: Literal[
        "fleet.model_builder.featurizers.MoleculeFeaturizer"
    ] = "fleet.model_builder.featurizers.MoleculeFeaturizer"
    name: str

    constructor_args: FleetmoleculefeaturizerConstructorArgs

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

    forward_args: FleetmoleculefeaturizerForwardArgsReferences


class FleetintegerfeaturizerConstructorArgsSummary(BaseModel):
    """
    Summarizes what types are the arguments necessary to instantiate fleet.model_builder.featurizers.IntegerFeaturizer

    Generated code
    """


class FleetintegerfeaturizerForwardArgsSummary(BaseModel):
    """
    Maps to references for the the arguments of fleet.model_builder.featurizers.IntegerFeaturizer.forward or fleet.model_builder.featurizers.IntegerFeaturizer.__call__

    References can be names of layers/featurizers defined in the same model config or reference an attribute
    of the same component

    Generated code
    """

    input_ = "<class 'str'>"


class FleetintegerfeaturizerForwardArgsReferences(BaseModel):

    input_: str


class FleetintegerfeaturizerSummary(CamelCaseModel):
    type: Literal[
        "fleet.model_builder.featurizers.IntegerFeaturizer"
    ] = "fleet.model_builder.featurizers.IntegerFeaturizer"
    constructor_args_summary = FleetintegerfeaturizerConstructorArgsSummary()

    forward_args_summary = FleetintegerfeaturizerForwardArgsSummary()


@options_manager.config_featurizer(summary_cls=FleetintegerfeaturizerSummary)
class FleetintegerfeaturizerLayerConfig(CamelCaseModel):
    """
    Layer configuration.

    "type" is a discriminator field, and each possible value for it the
    args type will be mapped to the arguments of the respective class

    Generated code
    """

    type: Literal[
        "fleet.model_builder.featurizers.IntegerFeaturizer"
    ] = "fleet.model_builder.featurizers.IntegerFeaturizer"
    name: str

    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
            return lib_cls
        return lib_cls()

    forward_args: FleetintegerfeaturizerForwardArgsReferences


class FleetdnasequencefeaturizerConstructorArgsSummary(BaseModel):
    """
    Summarizes what types are the arguments necessary to instantiate fleet.model_builder.featurizers.DNASequenceFeaturizer

    Generated code
    """


class FleetdnasequencefeaturizerForwardArgsSummary(BaseModel):
    """
    Maps to references for the the arguments of fleet.model_builder.featurizers.DNASequenceFeaturizer.forward or fleet.model_builder.featurizers.DNASequenceFeaturizer.__call__

    References can be names of layers/featurizers defined in the same model config or reference an attribute
    of the same component

    Generated code
    """

    input_ = "<class 'str'>"


class FleetdnasequencefeaturizerForwardArgsReferences(BaseModel):

    input_: str


class FleetdnasequencefeaturizerSummary(CamelCaseModel):
    type: Literal[
        "fleet.model_builder.featurizers.DNASequenceFeaturizer"
    ] = "fleet.model_builder.featurizers.DNASequenceFeaturizer"
    constructor_args_summary = (
        FleetdnasequencefeaturizerConstructorArgsSummary()
    )

    forward_args_summary = FleetdnasequencefeaturizerForwardArgsSummary()


@options_manager.config_featurizer(
    summary_cls=FleetdnasequencefeaturizerSummary
)
class FleetdnasequencefeaturizerLayerConfig(CamelCaseModel):
    """
    Layer configuration.

    "type" is a discriminator field, and each possible value for it the
    args type will be mapped to the arguments of the respective class

    Generated code
    """

    type: Literal[
        "fleet.model_builder.featurizers.DNASequenceFeaturizer"
    ] = "fleet.model_builder.featurizers.DNASequenceFeaturizer"
    name: str

    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
            return lib_cls
        return lib_cls()

    forward_args: FleetdnasequencefeaturizerForwardArgsReferences


class FleetrnasequencefeaturizerConstructorArgsSummary(BaseModel):
    """
    Summarizes what types are the arguments necessary to instantiate fleet.model_builder.featurizers.RNASequenceFeaturizer

    Generated code
    """


class FleetrnasequencefeaturizerForwardArgsSummary(BaseModel):
    """
    Maps to references for the the arguments of fleet.model_builder.featurizers.RNASequenceFeaturizer.forward or fleet.model_builder.featurizers.RNASequenceFeaturizer.__call__

    References can be names of layers/featurizers defined in the same model config or reference an attribute
    of the same component

    Generated code
    """

    input_ = "<class 'str'>"


class FleetrnasequencefeaturizerForwardArgsReferences(BaseModel):

    input_: str


class FleetrnasequencefeaturizerSummary(CamelCaseModel):
    type: Literal[
        "fleet.model_builder.featurizers.RNASequenceFeaturizer"
    ] = "fleet.model_builder.featurizers.RNASequenceFeaturizer"
    constructor_args_summary = (
        FleetrnasequencefeaturizerConstructorArgsSummary()
    )

    forward_args_summary = FleetrnasequencefeaturizerForwardArgsSummary()


@options_manager.config_featurizer(
    summary_cls=FleetrnasequencefeaturizerSummary
)
class FleetrnasequencefeaturizerLayerConfig(CamelCaseModel):
    """
    Layer configuration.

    "type" is a discriminator field, and each possible value for it the
    args type will be mapped to the arguments of the respective class

    Generated code
    """

    type: Literal[
        "fleet.model_builder.featurizers.RNASequenceFeaturizer"
    ] = "fleet.model_builder.featurizers.RNASequenceFeaturizer"
    name: str

    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
            return lib_cls
        return lib_cls()

    forward_args: FleetrnasequencefeaturizerForwardArgsReferences


class FleetproteinsequencefeaturizerConstructorArgsSummary(BaseModel):
    """
    Summarizes what types are the arguments necessary to instantiate fleet.model_builder.featurizers.ProteinSequenceFeaturizer

    Generated code
    """


class FleetproteinsequencefeaturizerForwardArgsSummary(BaseModel):
    """
    Maps to references for the the arguments of fleet.model_builder.featurizers.ProteinSequenceFeaturizer.forward or fleet.model_builder.featurizers.ProteinSequenceFeaturizer.__call__

    References can be names of layers/featurizers defined in the same model config or reference an attribute
    of the same component

    Generated code
    """

    input_ = "<class 'str'>"


class FleetproteinsequencefeaturizerForwardArgsReferences(BaseModel):

    input_: str


class FleetproteinsequencefeaturizerSummary(CamelCaseModel):
    type: Literal[
        "fleet.model_builder.featurizers.ProteinSequenceFeaturizer"
    ] = "fleet.model_builder.featurizers.ProteinSequenceFeaturizer"
    constructor_args_summary = (
        FleetproteinsequencefeaturizerConstructorArgsSummary()
    )

    forward_args_summary = FleetproteinsequencefeaturizerForwardArgsSummary()


@options_manager.config_featurizer(
    summary_cls=FleetproteinsequencefeaturizerSummary
)
class FleetproteinsequencefeaturizerLayerConfig(CamelCaseModel):
    """
    Layer configuration.

    "type" is a discriminator field, and each possible value for it the
    args type will be mapped to the arguments of the respective class

    Generated code
    """

    type: Literal[
        "fleet.model_builder.featurizers.ProteinSequenceFeaturizer"
    ] = "fleet.model_builder.featurizers.ProteinSequenceFeaturizer"
    name: str

    def create(self):
        lib_cls = get_class_from_path_string(self.type)
        if is_func(lib_cls):
            return lib_cls
        return lib_cls()

    forward_args: FleetproteinsequencefeaturizerForwardArgsReferences


LayersType = Annotated[
    Union[
        FleetonehotLayerConfig,
        FleetglobalpoolingLayerConfig,
        FleetconcatLayerConfig,
        FleetaddpoolingLayerConfig,
        TorchlinearLayerConfig,
        TorchsigmoidLayerConfig,
        TorchreluLayerConfig,
        TorchgeometricgcnconvLayerConfig,
        TorchembeddingLayerConfig,
        TorchtransformerencoderlayerLayerConfig,
    ],
    Field(discriminator="type"),
]

FeaturizersType = Annotated[
    Union[
        FleetmoleculefeaturizerLayerConfig,
        FleetintegerfeaturizerLayerConfig,
        FleetdnasequencefeaturizerLayerConfig,
        FleetrnasequencefeaturizerLayerConfig,
        FleetproteinsequencefeaturizerLayerConfig,
    ],
    Field(discriminator="type"),
]

LayersArgsType = Annotated[
    Union[
        FleetonehotSummary,
        FleetglobalpoolingSummary,
        FleetconcatSummary,
        FleetaddpoolingSummary,
        TorchlinearSummary,
        TorchsigmoidSummary,
        TorchreluSummary,
        TorchgeometricgcnconvSummary,
        TorchembeddingSummary,
        TorchtransformerencoderlayerSummary,
    ],
    Field(discriminator="type"),
]

FeaturizersArgsType = Annotated[
    Union[
        FleetmoleculefeaturizerSummary,
        FleetintegerfeaturizerSummary,
        FleetdnasequencefeaturizerSummary,
        FleetrnasequencefeaturizerSummary,
        FleetproteinsequencefeaturizerSummary,
    ],
    Field(discriminator="type"),
]
