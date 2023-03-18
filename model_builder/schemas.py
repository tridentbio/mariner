"""
Object schemas used by the model builder
"""
# Temporary file to hold all extracted mariner schemas
from typing import Any, List, Literal, Optional, Union

import networkx as nx
import yaml
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    root_validator,
    validator,
)
from typing_extensions import Annotated

from model_builder import generate
from model_builder.components_query import (
    get_component_constructor_args_by_type,
)
from model_builder.layers_schema import FeaturizersType, LayersType
from model_builder.utils import CamelCaseModel, get_references_dict


class NumericalDataType(CamelCaseModel):
    """
    Data type for a numerical series
    """

    domain_kind: Literal["numeric"] = Field("numeric")

    @validator("domain_kind")
    def check_domain_kind(cls, value: Any):
        """Validates domain_kind"""
        return "numeric"


class QuantityDataType(NumericalDataType):
    """
    Data type for a numerical series bound to a unit
    """

    unit: str


class StringDataType(CamelCaseModel):
    """
    Data type for series of strings
    """

    domain_kind: Literal["string"] = Field("string")

    @validator("domain_kind")
    def check_domain_kind(cls, value: Any):
        """Validates domain_kind"""
        return "string"


class CategoricalDataType(CamelCaseModel):
    """
    Data type for a series of categorical column
    """

    domain_kind: Literal["categorical"] = Field("categorical")
    classes: dict[Union[str, int], int]

    @validator("domain_kind")
    def check_domain_kind(cls, value: Any):
        """Validates domain_kind"""
        return "categorical"


class SmileDataType(CamelCaseModel):
    """
    Data type for a series of SMILEs strings column
    """

    domain_kind: Literal["smiles"] = Field("smiles")

    @validator("domain_kind")
    def check_domain_kind(cls, value: Any):
        """Validates domain_kind"""
        return "smiles"


class DNADataType(CamelCaseModel):
    """
    Data type for a series of DNA strings column
    """

    domain_kind: Literal["dna"] = Field("dna")

    @validator("domain_kind")
    def check_domain_kind(cls, value: Any):
        """Validates domain_kind"""
        return "dna"


class RNADataType(CamelCaseModel):
    """
    Data type for a series of RNA strings column
    """

    domain_kind: Literal["rna"] = Field("rna")

    @validator("domain_kind")
    def check_domain_kind(cls, value: Any):
        """Validates domain_kind"""
        return "rna"


class ProteinDataType(CamelCaseModel):
    """
    Data type for a series of protein strings column
    """

    domain_kind: Literal["protein"] = Field("protein")

    @validator("domain_kind")
    def check_domain_kind(cls, value: Any):
        """Validates domain_kind"""
        return "protein"


# TODO: make data_type optional
class ColumnConfig(CamelCaseModel):
    """
    Describes a column based on its data type and index
    """

    name: str
    data_type: Union[
        QuantityDataType,
        NumericalDataType,
        StringDataType,
        SmileDataType,
        CategoricalDataType,
        DNADataType,
        RNADataType,
        ProteinDataType,
    ] = Field(...)


class TargetConfig(ColumnConfig):
    """
    Describes a target column based on its data type and index
    """

    out_module: Optional[str]
    loss_fn: Optional[str]
    column_type: Optional[Literal["regression", "multiclass", "binary"]]


class UnknownComponentType(ValueError):
    """
    Raised when an unknown component type is detected

    Attributes:
        component_name: The id of the component with bad type
    """

    component_name: str

    def __init__(self, *args, component_name: str):
        super().__init__(*args)
        self.component_name = component_name


class MissingComponentArgs(ValueError):
    """
    Raised when there are missing arguments for a component.

    It's used by the frontend editor to provide accurate user feedback
    on what layer/featurizer went wrong (using the layer/featurizer id instead of json
    location)

    Attributes:
        component_name: component id that failed
        missing: list of fields that are missing
    """

    component_name: str
    missing: List[Union[str, int]]

    def __init__(self, *args, missing: List[Union[str, int]], component_name: str):
        super().__init__(*args)
        self.component_name = component_name
        self.missing = missing


AnnotatedLayersType = Annotated[LayersType, Field(discriminator="type")]
AnnotatedFeaturizersType = Annotated[FeaturizersType, Field(discriminator="type")]
LossType = Literal[
    "torch.nn.MSELoss", "torch.nn.CrossEntropyLoss", "torch.nn.BCEWithLogitsLoss"
]

ALLOWED_CLASSIFIER_LOSSES = ["torch.nn.CrossEntropyLoss", "torch.nn.BCEWithLogitsLoss"]
ALLOWED_REGRESSOR_LOSSES = ["torch.nn.MSELoss"]
DEFAULT_LOSS_MAP = {
    "regression": "torch.nn.MSELoss",
    "binary": "torch.nn.BCEWithLogitsLoss",
    "multiclass": "torch.nn.CrossEntropyLoss",
}


def is_regression(target_column: TargetConfig):
    """
    Returns ``True`` if ``target_column`` is numeric and therefore the model
    that predicts it is a regressor
    """
    if not target_column.column_type:
        return target_column.data_type.domain_kind == "numeric"
    return target_column.column_type == "regression"


def is_classifier(target_column: TargetConfig):
    """
    Returns ``True`` if the ``target_column`` is categorical and therefore the model
    that predicts it is a classifier
    """
    if not target_column.column_type:
        return target_column.data_type.domain_kind == "categorical"
    return target_column.column_type in ["binary", "multiclass"]


def is_binary(target_column: ColumnConfig):
    """
    Returns ``True`` if the ``target_column`` is categorical and has only 2 classes
    """
    return is_classifier(target_column) and len(target_column.data_type.classes) == 2


def infer_column_type(column: TargetConfig):
    """
    Infer column type based on its data type
    """
    if is_regression(column):
        return "regression"
    if is_classifier(column):
        return "multiclass" if len(column.data_type.classes) > 2 else "binary"
    raise ValueError(f"Unknown column type for {column.name}")


# TODO: move featurizers to feature_columns
class DatasetConfig(CamelCaseModel):
    """
    Describes a dataset for the model in terms of it's used columns
    """

    name: str
    target_columns: List[TargetConfig]
    feature_columns: List[ColumnConfig]


class ModelSchema(CamelCaseModel):
    """
    A serializable neural net architecture.
    """

    @root_validator(pre=True)
    def check_types_defined(cls, values):
        """Pydantic validator that checks if layers and featurizer types are
        known and secure (it's from one of the trusted 3rd party ML libs)

        Args:
            values (dict): dictionary with object values

        Raises:
            UnknownComponentType: in case a layer of featurizer has unknown ``type``
        """
        layers: List[AnnotatedLayersType] = values.get("layers")
        featurizers: List[AnnotatedFeaturizersType] = values.get("featurizers")
        layer_types = [layer.name for layer in generate.layers]
        for layer in layers:
            if not isinstance(layer, dict):
                layer = layer.dict()
            if layer["type"] not in layer_types:
                raise UnknownComponentType(
                    "A layer has unknown type", component_name=layer["name"]
                )
        if featurizers:
            featurizer_types = [featurizer.name for featurizer in generate.featurizers]
            for featurizer in featurizers:
                if not isinstance(featurizer, dict):
                    featurizer = featurizer.dict()
                if featurizer["type"] not in featurizer_types:
                    raise UnknownComponentType(
                        f"A featurizer has unknown type: {featurizer['type']}",
                        component_name=featurizer["name"],
                    )
        return values

    @root_validator(pre=True)
    def check_no_missing_args(cls, values):
        """Pydantic validator to check component arguments

        Checks if all layers and featurizer have the nessery arguments

        Args:
            values (dict): dict with object values

        Raises:
            MissingComponentArgs: if some component is missing required args
        """
        layers: List[AnnotatedLayersType] = values.get("layers")
        featurizers: List[AnnotatedFeaturizersType] = values.get("featurizers")
        errors = []
        for layer in layers:
            if not isinstance(layer, dict):
                layer = layer.dict()
            args_cls = get_component_constructor_args_by_type(layer["type"])
            if not args_cls or "constructorArgs" not in layer:
                continue
            try:
                args_cls.validate(layer["constructorArgs"])
            except ValidationError as exp:
                errors += [
                    MissingComponentArgs(
                        missing=[missing_arg_name for missing_arg_name in error["loc"]],
                        component_name=layer["name"],
                    )
                    for error in exp.errors()
                    if error["type"] == "value_error.missing"
                ]
        if featurizers:
            for featurizer in featurizers:
                if not isinstance(featurizer, dict):
                    featurizer = featurizer.dict()
                args_cls = get_component_constructor_args_by_type(featurizer["type"])
                if not args_cls or "constructorArgs" not in featurizer:
                    continue
                try:
                    args_cls.validate(featurizer["constructorArgs"])
                except ValidationError as exp:
                    errors += [
                        MissingComponentArgs(
                            missing=[
                                missing_arg_name for missing_arg_name in error["loc"]
                            ],
                            component_name=featurizer["name"],
                        )
                        for error in exp.errors()
                        if error["type"] == "value_error.missing"
                    ]

        if len(errors) > 0:
            # TODO: raise all errors grouped in a single validation error
            raise errors[0]
        return values

    @root_validator(pre=True)
    def autofill_loss_fn(cls, values: dict) -> Any:
        """Validates or infer the loss_fn attribute

        Automatically fills and validates the loss_fn field based on the target_column
        of the dataset.target_column field

        Args:
            value: user given value for loss_fn
            values: values of the model schema


        Raises:
            ValidationError: if the loss_fn is invalid for the defined task and target_columns
            ValueError: if the loss_fn could not be inferred
        """

        if not values.get("dataset") or not values.get("layers"):
            raise ValidationError("You must specify dataset and layers")

        dataset = (
            values["dataset"]
            if isinstance(values["dataset"], DatasetConfig)
            else DatasetConfig(**values["dataset"])
        )

        layers: List[AnnotatedLayersType] = values["layers"]

        try:
            for i, target_column in enumerate(dataset.target_columns):
                if not target_column.column_type:
                    target_column.column_type = infer_column_type(target_column)

                if not target_column.loss_fn:
                    target_column.loss_fn = DEFAULT_LOSS_MAP[target_column.column_type]

                if not target_column.out_module:
                    if len(dataset.target_columns) > 1:
                        raise ValidationError(
                            "You must specify out_module for each target column."
                        )
                    target_column.out_module = (
                        layers[-1]["name"]
                        if isinstance(layers[-1], dict)
                        else layers[-1].name
                    )

                assert (
                    target_column.column_type == "regression"
                    and target_column.loss_fn in ALLOWED_REGRESSOR_LOSSES
                ) or (
                    target_column.column_type in ("multiclass", "binary")
                    and target_column.loss_fn in ALLOWED_CLASSIFIER_LOSSES
                )

                dataset.target_columns[i] = target_column

            return {
                **values,
                "dataset": dataset.dict(),
            }
        except KeyError:
            raise ValueError(
                f"Can't determine task for output {target_column}."
                f"Column type {target_column.column_type} have not default loss function."
            )
        except AssertionError:
            raise ValidationError(
                f"Loss function {target_column.loss_fn} is not valid for {target_column.column_type} task"
            )

    name: str
    dataset: DatasetConfig
    layers: List[AnnotatedLayersType] = []
    featurizers: List[AnnotatedFeaturizersType] = []

    def make_graph(self):
        """Makes a graph of the layers and featurizers

        The graph is used for a topological walk on the schema
        """
        g = nx.DiGraph()
        for feat in self.featurizers:
            g.add_node(feat.name)
        for layer in self.layers:
            g.add_node(layer.name)

        def _to_dict(arr: List[BaseModel]):
            return [item.dict() for item in arr]

        layers_and_featurizers = _to_dict(self.layers)
        if self.featurizers:
            layers_and_featurizers += _to_dict(self.featurizers)
        for feat in layers_and_featurizers:
            if "forward_args" not in feat:
                continue
            references = get_references_dict(feat["forward_args"])
            for key, value in references.items():
                if not value:  # optional forward_arg that was not set
                    continue
                if isinstance(value, str):
                    reference = value.split(".")[0]
                    g.add_edge(reference, feat["name"], attr=key)
                elif isinstance(value, list):
                    for reference in value:
                        reference = reference.split(".")[0]
                        g.add_edge(reference, feat["name"], attr=key)

                else:
                    continue
        return g

    @classmethod
    def from_yaml(cls, yamlstr):
        """Parses a model schema object directly form a yaml str

        Args:
            yamlstr (str): yaml str
        """
        config_dict = yaml.safe_load(yamlstr)
        return ModelSchema(**config_dict)


if __name__ == "__main__":
    print(ModelSchema.schema_json())
