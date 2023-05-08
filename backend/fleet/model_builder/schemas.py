"""
Object schemas used by the model builder
"""
# Temporary file to hold all extracted mariner schemas
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import networkx as nx
from pydantic import BaseModel, Field, root_validator
from typing_extensions import Annotated

from fleet.dataset_schemas import ColumnConfig, DatasetConfig
from fleet.model_builder import generate
from fleet.model_builder.components_query import (
    get_component_constructor_args_by_type,
)
from fleet.model_builder.layers_schema import (
    FeaturizersArgsType,
    LayersArgsType,
    LayersType,
)
from fleet.model_builder.utils import CamelCaseModel, get_references_dict
from fleet.yaml_model import YAML_Model


class TargetConfig(ColumnConfig):
    """
    Describes a target column based on its data type and index
    """

    out_module: str
    loss_fn: Optional[str] = None
    column_type: Optional[Literal["regression", "multiclass", "binary"]] = None


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
LossType = Literal[
    "torch.nn.MSELoss", "torch.nn.CrossEntropyLoss", "torch.nn.BCEWithLogitsLoss"
]
AllowedLossesType = List[Dict[str, str]]


class AllowedLosses(CamelCaseModel):
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

    @property
    def type_map(self):
        return {"regression": "regr", "binary": "bin_class", "multiclass": "mc_class"}

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


def is_binary(target_column: TargetConfig):
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
class TorchDatasetConfig(DatasetConfig):
    """
    Describes a dataset for the model in terms of it's used columns
    """

    target_columns: Sequence[TargetConfig]

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
                print("LOSS NOT ALLOWED")
                raise ValueError(
                    f"Loss function is not valid for  task",
                )

            values["target_columns"][i] = target_column

        return values


class TorchModelSchema(CamelCaseModel, YAML_Model):
    """
    A serializable neural net architecture.
    """

    layers: List[AnnotatedLayersType] = []

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
        layer_types = [layer.name for layer in generate.layers]
        for layer in layers:
            if not isinstance(layer, dict):
                layer = layer.dict()
            if layer["type"] not in layer_types:
                print("Unknown type")
                raise UnknownComponentType(
                    "A layer has unknown type",
                    component_name=layer["name"],
                )

        return values

    @root_validator(pre=True)
    def check_no_missing_args(cls, values):
        """Pydantic validator to check component arguments

        Checks if all layers and featurizer have the necessary arguments

        Args:
            values (dict): dict with object values

        Raises:
            MissingComponentArgs: if some component is missing required args
        """
        layers: List[AnnotatedLayersType] = values.get("layers")
        errors = []
        for layer in layers:
            if not isinstance(layer, dict):
                layer = layer.dict()
            args_cls = get_component_constructor_args_by_type(layer["type"])
            if not args_cls or "constructorArgs" not in layer:
                continue
            try:
                args_cls.validate(layer["constructorArgs"])
            except ValueError as exp:
                errors += [
                    MissingComponentArgs(
                        missing=[missing_arg_name for missing_arg_name in error["loc"]],
                        component_name=layer["name"],
                    )
                    for error in exp.errors()
                    if error["type"] == "value_error.missing"
                ]

        if len(errors) > 0:
            # TODO: raise all errors grouped in a single validation error
            print("errors...")
            raise errors[0]
        return values

    def make_graph(self):
        """Makes a graph of the layers and featurizers

        The graph is used for a topological walk on the schema
        """
        g = nx.DiGraph()
        for layer in self.layers:
            g.add_node(layer.name)

        def _to_dict(arr: List[BaseModel]):
            return [item.dict() for item in arr]

        layers_and_featurizers = _to_dict(self.layers)
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


class ComponentAnnotation(CamelCaseModel):
    """
    Gives extra information about the layer/featurizer
    """

    docs_link: Optional[str]
    docs: Optional[str]
    output_type: Optional[str]
    class_path: str
    type: Literal["featurizer", "layer"]


class ComponentOption(ComponentAnnotation):
    """
    Describes an option to be used in the ModelSchema.layers or ModelSchema.featurizers
    """

    component: Union[LayersArgsType, FeaturizersArgsType]
    default_args: Optional[Dict[str, Any]] = None
    args_options: Optional[Dict[str, List[str]]] = None


if __name__ == "__main__":
    print(TorchModelSchema.schema_json())
