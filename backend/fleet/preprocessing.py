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

+---------------------------+----------------------------------------------------+-----------------------------+------------------------------------------+------------+
| Name                      | Description                                        | Parameters                  | DataType                                 | Framework  |
+===========================+====================================================+=============================+============+==========================================+
| `FPVecFilteredTransformer`| A featurizer that transforms a SMILES string into  | `del_invariant`: bool       | :class:`fleet.data_types.SmilesDataType` | `molfeat`  |
|                           | a fingerprint vector.                              | `length`: int               |                                          |            |
+---------------------------+----------------------------------------------------+-----------------------------+------------------------------------------+------------------------+
| `DNASequenceFeaturizer`   | A featurizre that transformes a DNA string into a | `forward_args`: dict        | :class:`fleet.data_types.DNADataType`    | `fleet.model_builder.`  |
|                           | one hot encoded representation.                    |                             |                                          |            |
+---------------------------+----------------------------------------------------+-----------------------------+------------------------------------------+------------+
| `ProteinSequenceFeaturizer`| A featurizer that transformes a protein string    | `forward_args`: dict        | :class:`fleet.data_types.ProteinDataType`| `molfeat`  |
|                           | into a one hot encoded representation.             |                             |                                          |            |
+---------------------------+----------------------------------------------------+-----------------------------+------------------------------------------+------------+


Transforms
----------

Transforms are used to modify numeric data in a way that is useful for machine learning models.

+---------------------------+----------------------------------------------------+-----------------------------+------------------------------------------+------------+
| Name                      | Description                                        | Parameters                  | DataType                                 | Framework  |
+===========================+====================================================+=============================+============+==========================================+
| `LabelEncoder`            | A transform that encodes labels with value between | `forward_args`: dict        | :class:`fleet.data_types.LabelDataType`  | `sklearn`  |
|                           | 0 and n_classes-1.                                 |                             |                                          |            |
+---------------------------+----------------------------------------------------+-----------------------------+------------------------------------------+------------+
| `StandardScaler`          | A transform that standardizes features by removing | `with_mean`: bool           | :class:`fleet.data_types.NumericDataType`| `sklearn`  |
|                           | the mean and scaling to unit variance.             | `with_std`: bool            |                                          |            |
+---------------------------+----------------------------------------------------+-----------------------------+------------------------------------------+------------+


See Also:
    :mod:`fleet.utils.data`
    :mod:`fleet.model_builder.featurizers`
"""

from typing import Annotated, Dict, List, Literal, NewType, Union, get_args

from humps import camel
from pydantic import BaseModel, Field

from fleet.model_builder.layers_schema import (
    FeaturizersType as FeaturizersType_,
)
from fleet.model_builder.utils import get_class_from_path_string


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


class CreateFromType:
    """
    Adds a method to instantiate a class from it's class path (type) and constructor_args.

    Attributes:
        type (str): The class path of the class that will be instantiated.
        constructor_args (BaseModel): The constructor arguments passed to the class.
    """

    type: str
    constructor_args: Union[None, BaseModel] = None

    def create(self):
        """Creates an instance of the class from the class path and constructor_args."""
        class_ = get_class_from_path_string(self.type)
        if self.constructor_args:
            return class_(**self.constructor_args.dict())
        return class_()


# molfeat featurizers:
class FPVecFilteredTransformerConstructorArgs(BaseModel):
    """
    Models the constructor arguments of a FPVecFilteredTransformer.
    """

    del_invariant: Union[None, bool] = None
    length: Union[None, int] = None


class FPVecFilteredTransformerConfig(CamelCaseModel, CreateFromType):
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


class LabelEncoderConfig(CreateFromType, CamelCaseModel):
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


class OneHotEncoderConfig(CreateFromType, CamelCaseModel):
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


class StandardScalerConstructorArgs(BaseModel):
    """
    Models the constructor arguments of a sklearn.preprocessing.StandardScaler

    See also:
        :mod:`sklearn.preprocessing.StandardScaler`
    """

    with_mean: bool = True
    with_std: bool = True


class StandardScalerConfig(CreateFromType, CamelCaseModel):
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


# Custom Transformers


class NpConcatenateConfig(CreateFromType, CamelCaseModel):
    """
    Models the usage of numpy concatenate.
    """

    type: Literal[
        "fleet.model_builder.transforms.np_concatenate.NpConcatenate"
    ] = "fleet.model_builder.transforms.np_concatenate.NpConcatenate"
    name: str
    forward_args: Union[Dict[str, List[str]], List[str]]


TransformerType = NewType(
    "TransformerType",
    Annotated[  # type: ignore
        Union[
            StandardScalerConfig,
            LabelEncoderConfig,
            OneHotEncoderConfig,
            NpConcatenateConfig,
        ],
        Field(discriminator="type"),
    ],
)


FeaturizersType = NewType(
    "FeaturizersType",
    Annotated[
        Union[get_args(get_args(FeaturizersType_)[0]) + (FPVecFilteredTransformerConfig,)],  # type: ignore
        Field(discriminator="type"),
    ],
)


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
