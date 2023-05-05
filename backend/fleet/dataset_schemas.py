from typing import Annotated, Sequence, Union

from pydantic import Field

from fleet import data_types
from fleet.model_builder.layers_schema import FeaturizersType
from fleet.model_builder.utils import CamelCaseModel


class ColumnConfig(CamelCaseModel):
    """
    Describes a column based on its data type and index
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
    Describes a dataset for the model in terms of it's used columns
    """

    name: str
    target_columns: Sequence[ColumnConfig]
    feature_columns: Sequence[ColumnConfig]
    featurizers: Sequence[FeaturizersType] = []
