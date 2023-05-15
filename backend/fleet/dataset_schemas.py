"""
Classes used to describe datasets. They can be extended, but
it is not encouraged since it will required adapting the implementation
of some methods.
"""
from typing import Annotated, Sequence, Union

from pydantic import Field

from fleet import data_types
from fleet.model_builder.layers_schema import FeaturizersType
from fleet.model_builder.utils import CamelCaseModel


class ColumnConfig(CamelCaseModel):
    """
    Describes a column based on its data type and index.

    Attributes:
        name: The name of the column.
        data_type: One of :py:mod:`fleet.data_types`
    """

    name: str
    data_type: Union[
        data_types.QuantityDataType,
        data_types.NumericalDataType,
        data_types.StringDataType,
        data_types.SmileDataType,
        data_types.CategoricalDataType,
        data_types.DNADataType,
        data_types.RNADataType,
        data_types.ProteinDataType,
    ] = Field(...)


AnnotatedFeaturizersType = Annotated[FeaturizersType, Field(discriminator="type")]


class DatasetConfig(CamelCaseModel):
    """
    Describes a dataset for the model.

    Attributes:
        name: The dataset identifier.
        target_columns: A sequence of columns descriptions that should be use as
            targets by the ML algorithms.
        feature_columns: A sequence of column descriptions that should be use as
            features by the ML algorithms.
        featurizers: Allows to specify transformations of the columns in
            reproducible way.

    """

    name: str
    target_columns: Sequence[ColumnConfig]
    feature_columns: Sequence[ColumnConfig]
    featurizers: Sequence[FeaturizersType] = []
