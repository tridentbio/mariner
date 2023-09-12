"""
This package wraps the underlying preprocessing functions and schemas.
To define a preprocessing pipeline, torch dataset or preprocess a DataFrame
from the underlying schemas see :mod:`fleet.utils.data`.
Wrapped featurizers and transforms are gathered from :mod:`fleet.model_builder.featurizers`
and :mod:`sklearn.preprocessing`.

Featurizers
-----------

Featurizers are used to transform not numeric data into numeric matrices that
can be understood by machine learning models.

Transforms
----------

Transforms are used to modify numeric data in a way that is useful for machine learning models.

See Also:
    :mod:`fleet.utils.data`
    :mod:`fleet.model_builder.featurizers`

Todo:

    Remove the TransformerConfig and FeaturizerConfig classes and replace them with
    some alternative that doesn't use inheritance. This is because the inheritance
    is not used in the elsewhere code and is confusing.
"""

from typing import Annotated, Any, Dict, List, Literal, Union, get_args

from humps import camel
from pydantic import BaseModel, Field

from fleet.model_builder.layers_schema import (
    FeaturizersType as FeaturizersType_,
)
from fleet.model_builder.layers_schema import (
    FleetmoleculefeaturizerLayerConfig as FleetmoleculefeaturizerLayerConfig_,
)
from fleet.model_builder.utils import get_class_from_path_string
from fleet.options import options_manager


class CamelCaseModel(BaseModel):
    """
    Subclass this to work with camel case serialization of the model.
    """

    class Config:
        """
        Configures the model.
        """

        alias_generator = camel.case
        allow_population_by_field_name = True
        allow_population_by_alias = True
        underscore_attrs_are_private = True


class CreateFromType(CamelCaseModel):
    """
    Adds a method to instantiate a class from it's class path (type) and constructor_args.

    Attributes:
        type (str): The class path of the class that will be instantiated.
        constructor_args (BaseModel): The constructor arguments passed to the class.
    """

    type: str
    constructor_args: Union[BaseModel, dict[str, Any], None] = None

    def create(self):
        """Creates an instance of the class from the class path and constructor_args."""
        class_ = get_class_from_path_string(self.type)
        if self.constructor_args and isinstance(self.constructor_args, dict):
            return class_(**self.constructor_args)  # pylint: disable=E1134
        elif self.constructor_args and isinstance(
            self.constructor_args, BaseModel
        ):
            return class_(**self.constructor_args.dict())
        return class_()


class TransformConfigBase:
    """
    Base class for transforms.
    """

    def adapt_args_and_apply(self, method, args):
        """
        Method to perform fit, transform, fit_transform, inverse_transform, etc.
        Can be customized to normalize data after and before the method runs.
        """
        return method(*args)


class FleetmoleculefeaturizerLayerConfig(
    FleetmoleculefeaturizerLayerConfig_, TransformConfigBase
):
    """
    Includes the adapt_args_and_apply method to run the featurizer.
    """


class FPVecFilteredTransformerConstructorArgs(BaseModel):
    """
    Models the constructor arguments of a FPVecFilteredTransformer.
    """

    del_invariant: bool = False
    length: int = 2000


@options_manager.config_featurizer()
class FPVecFilteredTransformerConfig(CreateFromType, TransformConfigBase):
    """
    Models the usage of FPVecFilteredTransformer.
    """

    name: str
    constructor_args: FPVecFilteredTransformerConstructorArgs = (
        FPVecFilteredTransformerConstructorArgs()
    )
    type: Literal[
        "molfeat.trans.fp.FPVecFilteredTransformer"
    ] = "molfeat.trans.fp.FPVecFilteredTransformer"
    forward_args: dict


# sklearn transforms


@options_manager.config_featurizer()
class LabelEncoderConfig(CreateFromType, CamelCaseModel, TransformConfigBase):
    """
    Models the constructor arguments of a sklearn.preprocessing.LabelEncoder

    See also:
        :mod:`sklearn.preprocessing.LabelEncoder`
    """

    type: Literal[
        "sklearn.preprocessing.LabelEncoder"
    ] = "sklearn.preprocessing.LabelEncoder"
    name: str
    forward_args: Union[Dict[str, str], list[str]]

    def adapt_args_and_apply(self, method, args):
        args = map(lambda x: x.reshape(-1, 1), args)
        return super().adapt_args_and_apply(method, args)


@options_manager.config_featurizer()
class OneHotEncoderConfig(CreateFromType, CamelCaseModel, TransformConfigBase):
    """
    Models the constructor arguments of a sklearn.preprocessing.OneHotEncoder

    See also:
        :mod:`sklearn.preprocessing.OneHotEncoder`
    """

    type: Literal[
        "sklearn.preprocessing.OneHotEncoder"
    ] = "sklearn.preprocessing.OneHotEncoder"
    name: str
    forward_args: Union[Dict[str, str], List[str]]

    def adapt_args_and_apply(self, method, args):
        args = map(lambda x: x.reshape(-1, 1), args)
        result = super().adapt_args_and_apply(method, args)
        return result.toarray()


class StandardScalerConstructorArgs(BaseModel):
    """
    Models the constructor arguments of a sklearn.preprocessing.StandardScaler

    See also:
        :mod:`sklearn.preprocessing.StandardScaler`
    """

    with_mean: bool = True
    with_std: bool = True


@options_manager.config_transformer()
class StandardScalerConfig(
    CreateFromType, CamelCaseModel, TransformConfigBase
):
    """
    Models the usage of a StandardScaler.
    """

    type: Literal[
        "sklearn.preprocessing.StandardScaler"
    ] = "sklearn.preprocessing.StandardScaler"
    constructor_args: StandardScalerConstructorArgs = (
        StandardScalerConstructorArgs()
    )
    name: str
    forward_args: Union[Dict[str, str], list[str]]

    def adapt_args_and_apply(self, method, args):
        args = map(lambda x: x.reshape(-1, 1), args)
        return super().adapt_args_and_apply(method, args)


# Custom Transformers


class NpConcatenateConfig(CreateFromType, CamelCaseModel, TransformConfigBase):
    """
    Models the usage of numpy concatenate.
    """

    type: Literal[
        "fleet.model_builder.transforms.np_concatenate.NpConcatenate"
    ] = "fleet.model_builder.transforms.np_concatenate.NpConcatenate"
    name: str
    forward_args: Union[Dict[str, List[str]], List[str]]


TransformerType = Annotated[
    Union[
        StandardScalerConfig,
        NpConcatenateConfig,
    ],
    Field(discriminator="type"),
]


FeaturizersType = Annotated[
    Union[
        (OneHotEncoderConfig, LabelEncoderConfig, FPVecFilteredTransformerConfig, FleetmoleculefeaturizerLayerConfig)  # type: ignore
        + get_args(get_args(FeaturizersType_)[0])  # type: ignore
    ],
    Field(discriminator="type"),
]


class TransformConfig(CamelCaseModel):
    """
    Pydantic model to validate a transformer configuration.
    """

    __root__: TransformerType


class FeaturizerConfig(CamelCaseModel):
    """
    Pydantic model to validate a featurizer configuration.
    """

    __root__: FeaturizersType


__all__ = [
    "TransformConfig",
    "FeaturizerConfig",
    "FPVecFilteredTransformerConfig",
    "FleetmoleculefeaturizerLayerConfig",
    "LabelEncoderConfig",
    "OneHotEncoderConfig",
    "StandardScalerConfig",
    "NpConcatenateConfig",
]
