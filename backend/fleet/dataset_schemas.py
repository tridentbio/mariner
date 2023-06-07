"""
Classes used to describe datasets. They can be extended, but
it is not encouraged since it will required adapting the implementation
of some methods.
"""
from typing import Any, Dict, List, Literal, Optional, Union

from humps import camel
from pydantic import BaseModel, Field, root_validator

from fleet import data_types
from fleet.model_builder.layers_schema import FeaturizersType


class BaseDatasetModel(BaseModel):
    """
    Configures the models in this package.
    """

    class Config:
        """Configures the wrapper class to work as intended."""

        alias_generator = camel.case
        allow_population_by_field_name = True
        allow_population_by_alias = True
        underscore_attrs_are_private = True


class ColumnConfig(BaseDatasetModel):
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

    class Config:
        """Configures the wrapper class to work as intended."""

        alias_generator = camel.case
        allow_population_by_field_name = True
        allow_population_by_alias = True
        underscore_attrs_are_private = True


class TargetConfig(ColumnConfig):
    """
    Describes a target column based on its data type and index
    """

    out_module: str
    loss_fn: Optional[str] = None
    column_type: Optional[Literal["regression", "multiclass", "binary"]] = None


class DatasetConfig(BaseDatasetModel):
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
    target_columns: List[ColumnConfig]
    feature_columns: List[ColumnConfig]
    featurizers: List[FeaturizersType] = []


AllowedLossesType = List[Dict[str, str]]


class AllowedLosses(BaseDatasetModel):
    """
    List allowed losses for each column type
    """

    regr: AllowedLossesType = [{"key": "torch.nn.MSELoss", "value": "MSELoss"}]
    bin_class: AllowedLossesType = [
        {"key": "torch.nn.BCEWithLogitsLoss", "value": "BCEWithLogitsLoss"}
    ]
    mc_class: AllowedLossesType = [
        {"key": "torch.nn.CrossEntropyLoss", "value": "CrossEntropyLoss"}
    ]
    type_map = {"regression": "regr", "binary": "bin_class", "multiclass": "mc_class"}

    def check(
        self, loss: str, column_type: Literal["regression", "multiclass", "binary"]
    ):
        """Check if loss is allowed for column_type"""
        allowed_losses_list = map(
            lambda x: x["key"], self.dict()[self.type_map[column_type]]
        )
        return loss in allowed_losses_list

    def get_default(
        self, column_type: Literal["regression", "multiclass", "binary"]
    ) -> str:
        """Get default loss for column_type"""
        return self.dict()[self.type_map[column_type]][0]["key"]


def is_regression(target_column: "TargetConfig"):
    """
    Returns ``True`` if ``target_column`` is numeric and therefore the model
    that predicts it is a regressor
    """
    if not target_column.column_type:
        return target_column.data_type.domain_kind == "numeric"
    return target_column.column_type == "regression"


def is_classifier(target_column: "TargetConfig"):
    """
    Returns ``True`` if the ``target_column`` is categorical and therefore the model
    that predicts it is a classifier
    """
    if not target_column.column_type:
        return target_column.data_type.domain_kind == "categorical"
    return target_column.column_type in ["binary", "multiclass"]


def is_binary(target_column: "TargetConfig"):
    """
    Returns ``True`` if the ``target_column`` is categorical and has only 2 classes
    """
    return is_classifier(target_column) and len(target_column.data_type.classes) == 2


def infer_column_type(column: "TargetConfig"):
    """
    Infer column type based on its data type
    """
    if is_regression(column):
        return "regression"
    if is_classifier(column):
        return "multiclass" if len(column.data_type.classes) > 2 else "binary"
    raise ValueError(f"Unknown column type for {column.name}")


class TorchDatasetConfig(DatasetConfig):
    """
    Describes a dataset for the model in terms of it's used columns
    """

    target_columns: List["TargetConfig"]

    @root_validator()
    def autofill_loss_fn(cls, values: dict) -> Any:
        """Validates or infer the loss_fn attribute

        Automatically fills and validates the loss_fn field based on the target_column
        of the dataset.target_column field

        Args:
            value: user given value for loss_fn
            values: values of the model schema


        Raises:
            ValueError: if the loss_fn is invalid for the defined task and
            target_columns
            ValueError: if the loss_fn could not be inferred
        """
        allowed_losses = AllowedLosses()
        target_columns = values.get("target_columns")
        if not target_columns:
            raise ValueError("Missing target_columns")
        for i, target_column in enumerate(target_columns):
            if not target_column.out_module:
                raise ValueError(
                    "You must specify out_module for each target column.",
                )
            if not target_column.column_type:
                target_column.column_type = infer_column_type(target_column)

            if not target_column.loss_fn:
                target_column.loss_fn = allowed_losses.get_default(
                    target_column.column_type
                )

            if not allowed_losses.check(
                target_column.loss_fn, target_column.column_type
            ):
                raise ValueError(
                    "Loss function is not valid for  task",
                )

            values["target_columns"][i] = target_column

        return values
