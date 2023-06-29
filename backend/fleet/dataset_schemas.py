"""
Classes used to describe datasets. They can be extended, but
it is not encouraged since it will required adapting the implementation
of some methods.
"""
from typing import Any, Dict, List, Literal, Optional, Union

from humps import camel
from pydantic import BaseModel, Field, root_validator

from fleet import data_types
from fleet.preprocessing import (
    CreateFromType,
    FeaturizerConfig,
    FeaturizersType,
    TransformConfig,
    TransformerType,
)
from fleet.yaml_model import YAML_Model


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
        data_types.NumericDataType,
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


class TargetTorchColumnConfig(ColumnConfig):
    """
    Describes a target column based on its data type and index
    """

    out_module: str
    loss_fn: Optional[str] = None
    column_type: Optional[Literal["regression", "multiclass", "binary"]] = None


class ColumnConfigWithPreprocessing(BaseDatasetModel):
    """
    Describes a column and it's preprocessing steps.
    """

    name: str
    data_type: data_types.DataType
    transforms: Union[None, List[CreateFromType]] = None
    featurizers: Union[None, List[CreateFromType]] = None


class DatasetConfig(BaseDatasetModel, YAML_Model):
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
    transforms: List[TransformerType] = []

    @property
    def columns(self):
        """
        Returns a list of all columns in the dataset.
        """
        return self.target_columns + self.feature_columns

    def get_column(self, col_name: str) -> ColumnConfig:
        """Get column by name"""
        for col in self.columns:
            if col.name == col_name:
                return col
        raise ValueError(f"Column {col_name} not found")


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
    type_map = {
        "regression": "regr",
        "binary": "bin_class",
        "multiclass": "mc_class",
    }

    def check(
        self,
        loss: str,
        column_type: Literal["regression", "multiclass", "binary"],
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


def is_regression(target_column: "TargetTorchColumnConfig"):
    """
    Returns ``True`` if ``target_column`` is numeric and therefore the model
    that predicts it is a regressor
    """
    if not target_column.column_type:
        return target_column.data_type.domain_kind == "numeric"
    return target_column.column_type == "regression"


def is_classifier(target_column: "TargetTorchColumnConfig"):
    """
    Returns ``True`` if the ``target_column`` is categorical and therefore the model
    that predicts it is a classifier
    """
    if not target_column.column_type:
        return target_column.data_type.domain_kind == "categorical"
    return target_column.column_type in ["binary", "multiclass"]


def is_binary(target_column: "TargetTorchColumnConfig"):
    """
    Returns ``True`` if the ``target_column`` is categorical and has only 2 classes
    """
    return is_classifier(target_column) and len(target_column.data_type.classes) == 2  # type: ignore


def infer_column_type(column: "TargetTorchColumnConfig"):
    """
    Infer column type based on its data type
    """
    if is_regression(column):
        return "regression"
    if is_classifier(column):
        return "multiclass" if len(column.data_type.classes) > 2 else "binary"  # type: ignore
    raise ValueError(f"Unknown column type for {column.name}")


class TorchDatasetConfig(DatasetConfig):
    """
    Describes a dataset for the model in terms of it's used columns
    """

    target_columns: List["TargetTorchColumnConfig"]

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


class DatasetConfigWithPreprocessing(BaseDatasetModel, YAML_Model):
    """
    Describes a dataset configuration with preprocessing steps.
    """

    name: str
    target_columns: List[ColumnConfigWithPreprocessing]
    feature_columns: List[ColumnConfigWithPreprocessing]

    def to_dataset_config(self) -> DatasetConfig:
        featurizers = []
        transforms = []
        for col in self.feature_columns + self.target_columns:

            def f(array, col, attr, parser):
                if hasattr(col, attr) and isinstance(getattr(col, attr), list):
                    previous_key = col.name
                    for idx, col_featurizer in enumerate(getattr(col, attr)):
                        key = f"{col.name}-feat-{idx}"
                        featurizer_args = col_featurizer.dict() | {
                            "name": key,
                            "forward_args": [f"${previous_key}"],
                        }
                        if featurizer_args["constructor_args"] is None:
                            featurizer_args.pop("constructor_args")
                        array.append(parser.parse_obj(featurizer_args))
                        previous_key = key

            f(featurizers, col, "featurizers", FeaturizerConfig)
            f(transforms, col, "transforms", TransformConfig)

        return DatasetConfig(
            name=self.name,
            target_columns=[
                ColumnConfig(name=col.name, data_type=col.data_type)
                for col in self.target_columns
            ],
            feature_columns=[
                ColumnConfig(name=col.name, data_type=col.data_type)
                for col in self.feature_columns
            ],
            featurizers=[feat.__root__ for feat in featurizers],
            transforms=[transf.__root__ for transf in transforms],
        )


class FeaturizerWrapper(BaseModel):
    """
    Wraps a featurizer so it's union type can be validated.

    Attributes:
        featurizer: The featurizer.
    """

    featurizer: FeaturizersType


class FeaturizerBuilder:
    """
    Builds a featurizer.

    Attributes:
        name: The name of the featurizer.
        class_path: The class path of the featurizer.
    """

    def __init__(self, name: str, class_path: str):
        self.name = name
        self.class_path = class_path
        self.constructor_args = {}
        self.forward_args = {}

    def with_constructor_args(self, **kwargs):
        """
        Sets the constructor arguments of the featurizer.

        Args:
            **kwargs: The constructor arguments.
        """
        self.constructor_args = kwargs
        return self

    def with_forward_args(self, **kwargs):
        """
        Sets the forward arguments of the featurizer.

        Args:
            **kwargs: The forward arguments.
        """
        self.forward_args = kwargs
        return self

    def build(self):
        """
        Builds and validates the featurizer.
        """
        wrapper = FeaturizerWrapper.parse_obj(
            {
                "featurizer": {
                    "name": self.name,
                    "class_path": self.class_path,
                    "constructor_args": self.constructor_args,
                    "forward_args": self.forward_args,
                }
            }
        )
        return wrapper.featurizer


class DatasetConfigBuilder:
    """
    Builds a DatasetConfig.
    """

    def __init__(self, name: Union[None, str] = None):
        self.name = name
        self.target_columns = []
        self.feature_columns = []
        self.featurizers = []
        self.transforms = []

    def with_name(self, name: str) -> "DatasetConfigBuilder":
        """
        Sets the name of the dataset.

        Args:
            name: The name of the dataset.

        Returns:
            The builder.
        """
        self.name = name
        return self

    def with_targets(self, out_module: str = "", **kwargs):
        """
        Sets the target columns of the dataset.

        Each keyword argument should be a column name and a data type.

        Args:
            **kwargs: The columns to add to the dataset.

        Returns:
            The builder.
        """
        self.target_columns = [
            TargetTorchColumnConfig(
                name=name, out_module=out_module, data_type=data_type
            )
            for name, data_type in kwargs.items()
        ]
        return self

    def with_features(self, **kwargs):
        """
        Sets the feature columns of the dataset.

        Each keyword argument should be a column name and a data type.

        Args:
            **kwargs: The columns to add to the dataset.

        Returns:
            The builder.
        """
        self.feature_columns = [
            ColumnConfig(name=name, data_type=data_type)
            for name, data_type in kwargs.items()
        ]
        return self

    def add_featurizers(self, featurizer: FeaturizersType):
        """
        Adds a featurizer on the dataset.
        """
        self.featurizers.append(featurizer)
        return self

    def add_transforms(self, transform: Any):
        """
        Adds a featurizer on the dataset.
        """
        self.transforms.append(transform)
        return self

    def build_torch(self) -> TorchDatasetConfig:
        """
        Builds the dataset configuration to use with torch models.

        Returns:
            The dataset configuration.
        """
        self._validate()
        return TorchDatasetConfig(
            name=self.name,  # type:ignore
            target_columns=[
                TargetTorchColumnConfig(**target_col.dict()) for target_col in self.target_columns  # type: ignore
            ],
            feature_columns=self.feature_columns,
            featurizers=self.featurizers,
            transforms=self.transforms,
        )

    def _validate(self):
        if not self.name:
            raise ValueError("Missing name for dataset")

    def build(self) -> DatasetConfig:
        """
        Builds the dataset configuration.

        Returns:
            The dataset configuration.
        """
        self._validate()
        print(
            self.name,
            self.target_columns + self.feature_columns,
            self.featurizers,
            self.transforms,
        )
        return DatasetConfig(
            name=self.name,  # type: ignore
            target_columns=[
                ColumnConfig(**target_col.dict()) for target_col in self.target_columns  # type: ignore
            ],
            feature_columns=self.feature_columns,
            featurizers=self.featurizers,
            transforms=self.transforms,
        )


def is_regression_column(
    dataset_config: Union[DatasetConfig, TorchDatasetConfig], column_name: str
) -> bool:
    """Checks if a model trained on ``dataset_config`` predicting values
    for ``column_name`` should be a regressor.

    This function is used to determine if a column is a regression column,
    and work on all supported dataset configs.

    Args:
        dataset_config: The dataset configuration to check if the column is a regression column.
        column_name: The name of the column to check.

    Returns:
        bool: True if the column is a regression column, False otherwise.

    Raises:
        ValueError: If the dataset_config is not a supported type.
    """
    if isinstance(dataset_config, TorchDatasetConfig):
        return is_regression(dataset_config.get_column(column_name))  # type: ignore
    elif isinstance(dataset_config, DatasetConfig):
        return (
            dataset_config.get_column(column_name).data_type.domain_kind
            == "numeric"
        )
    else:
        raise ValueError(
            f"dataset_config should be either DatasetConfig or TorchDatasetConfig, got {type(dataset_config)}"
        )


def is_categorical_column(
    dataset_config: Union[DatasetConfig, TorchDatasetConfig], column_name: str
) -> bool:
    """Checks if a model trained on ``dataset_config`` predicting values for
    ``column_name`` should be a classifier.

    This function is used to determine if a model trained on ``dataset_config``
    should be a classifier, and work on all supported dataset configs.

    Args:
        dataset_config: The dataset configuration to check if the column is a categorical column.
        column_name: The name of the column to check.

    Returns:
        bool: True if the column is a categorical column, False otherwise.

    Raises:
        ValueError: If the dataset_config is not a supported type.
    """
    if isinstance(dataset_config, TorchDatasetConfig):
        return is_classifier(dataset_config.get_column(column_name))  # type: ignore
    elif isinstance(dataset_config, DatasetConfig):
        return (
            dataset_config.get_column(column_name).data_type.domain_kind
            == "categorical"
        )
    else:
        raise ValueError(
            f"dataset_config should be either DatasetConfig or TorchDatasetConfig, got {type(dataset_config)}"
        )
