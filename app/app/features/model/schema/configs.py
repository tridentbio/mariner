from ast import literal_eval
from typing import Dict, List, Optional, Union, get_args, get_type_hints, Literal

import networkx as nx
import yaml
from pydantic import Field, ValidationError, root_validator

from app.features.dataset.schema import (
    CategoricalDataType,
    NumericalDataType,
    QuantityDataType,
    SmileDataType,
    StringDataType,
)
from app.features.model.components_query import get_component_args_by_type
from app.features.model.schema.layers_schema import (
    FeaturizersArgsType,
    FeaturizersType,
    LayersArgsType,
    LayersType,
)
from app.schemas.api import ApiBaseModel


# TODO: make data_type optional
class ColumnConfig(ApiBaseModel):
    name: str
    data_type: Union[
        QuantityDataType,
        NumericalDataType,
        StringDataType,
        SmileDataType,
        CategoricalDataType,
    ] = Field(...)


# TODO: move featurizers to feature_columns
class DatasetConfig(ApiBaseModel):
    name: str
    target_column: ColumnConfig
    feature_columns: List[ColumnConfig]


class ComponentAnnotation(ApiBaseModel):
    """
    Gives extra information about the layer/featurizer

    """

    docs_link: Optional[str]
    docs: Optional[str]
    positional_inputs: Dict[str, str]
    output_type: Optional[str]
    class_path: str
    type: Literal["featurizer", "layer"]


class ComponentOption(ComponentAnnotation):
    """
    Describes an option to be used in the ModelSchema.layers or ModelSchema.featurizers
    """

    component: Union[LayersArgsType, FeaturizersArgsType]


ModelOptions = List[ComponentOption]


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


class ModelSchema(ApiBaseModel):
    """
    A serializable model architecture


    """

    @root_validator(pre=True)
    def check_types_defined(cls, values):
        layers = values.get("layers")
        featurizers = values.get("featurizers")
        layer_types = [
            get_args(get_type_hints(cls)["type"])[0] for cls in get_args(LayersType)
        ]
        featurizer_types = [get_args(get_type_hints(FeaturizersType)["type"])[0]]
        for layer in layers:
            if not isinstance(layer, dict):
                layer = layer.dict()
            if layer["type"] not in layer_types:
                raise UnknownComponentType(
                    "A layer has unknown type", component_name=layer["name"]
                )
        for featurizer in featurizers:
            if not isinstance(featurizer, dict):
                featurizer = featurizer.dict()
            if featurizer["type"] not in featurizer_types:
                raise UnknownComponentType(
                    "A featurizer has unknown type", component_name=featurizer["name"]
                )
        return values

    @root_validator(pre=True)
    def check_no_missing_args(cls, values):
        layers = values.get("layers")
        featurizers = values.get("featurizers")
        errors = []
        for layer in layers:
            if not isinstance(layer, dict):
                layer = layer.dict()
            args_cls = get_component_args_by_type(layer["type"])
            if not args_cls:
                continue
            try:
                args_cls.validate(layer["args"])
            except ValidationError as exp:
                errors += [
                    MissingComponentArgs(
                        missing=[missing_arg_name for missing_arg_name in error["loc"]],
                        component_name=layer["name"],
                    )
                    for error in exp.errors()
                    if error["type"] == "value_error.missing"
                ]
        for featurizer in featurizers:
            if not isinstance(featurizer, dict):
                featurizer = featurizer.dict()
            args_cls = get_component_args_by_type(featurizer["type"])
            if not args_cls:
                continue
            try:
                args_cls.validate(featurizer["args"])
            except ValidationError as exp:
                errors += [
                    MissingComponentArgs(
                        missing=[missing_arg_name for missing_arg_name in error["loc"]],
                        component_name=featurizer["name"],
                    )
                    for error in exp.errors()
                    if error["type"] == "value_error.missing"
                ]

        if len(errors) > 0:
            raise errors[0]
        return values

    name: str
    dataset: DatasetConfig
    layers: List[LayersType]
    featurizers: List[FeaturizersType]

    def make_graph(self):
        g = nx.DiGraph()
        for feat in self.featurizers:
            g.add_node(feat.name)
        for layer in self.layers:
            g.add_node(layer.name)
        for layer in self.layers:
            if isinstance(layer.input, str):
                g.add_edge(layer.input, layer.name)
            else:
                for input_value in layer.input:
                    g.add_edge(input_value, layer.name)
        return g

    @classmethod
    def from_yaml(cls, yamlstr):
        config_dict = yaml.safe_load(yamlstr)
        return ModelSchema.parse_obj(config_dict)


if __name__ == "__main__":
    print(ModelSchema.schema_json())
