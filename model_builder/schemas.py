# Temporary file to hold all extracted mariner schemas
from typing import List, Literal, Union
from humps import camel

import networkx as nx
import yaml
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    root_validator,
    validator,
)
from model_builder import generate

from model_builder.components_query import get_component_constructor_args_by_type
from model_builder.layers_schema import (
    BaseLayerConfig,
    FeaturizersType,
    LayersType,
    PythonArgsBaseModel,
)


class CamelCaseModel(BaseModel):
    class Config:
        alias_generator = camel.case
        allow_population_by_field_name = True
        allow_population_by_alias = True
        underscore_attrs_are_private = True


class NumericalDataType(BaseModel):
    domain_kind: Literal["numeric"] = Field("numeric")

    @validator("domain_kind")
    def check_domain_kind(cls, v):
        return "numeric"


class QuantityDataType(NumericalDataType):
    unit: str


class StringDataType(BaseModel):
    domain_kind: Literal["string"] = Field("string")

    @validator("domain_kind")
    def check_domain_kind(cls, v):
        return "string"


class CategoricalDataType(BaseModel):
    domain_kind: Literal["categorical"] = Field("categorical")
    classes: dict[Union[str, int], int]

    @validator("domain_kind")
    def check_domain_kind(cls, v):
        return "categorical"


class SmileDataType(BaseModel):
    domain_kind: Literal["smiles"] = Field("smiles")

    @validator("domain_kind")
    def check_domain_kind(cls, v):
        return "smiles"


# TODO: make data_type optional
class ColumnConfig(CamelCaseModel):
    name: str
    data_type: Union[
        QuantityDataType,
        NumericalDataType,
        StringDataType,
        SmileDataType,
        CategoricalDataType,
    ] = Field(...)


# TODO: move featurizers to feature_columns
class DatasetConfig(CamelCaseModel):
    name: str
    target_column: ColumnConfig
    feature_columns: List[ColumnConfig]


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


def _unwrap_dollar(value: str) -> tuple[str, bool]:
    """
    Takes a string and remove it's reference indicators: $ or ${...}
    Returns the string unwrapped (if it was a reference) and `is_reference` boolean
    """
    if value.startswith("${") and value.endswith("}"):
        return value[2:-1], True
    elif value.startswith("$"):
        return value[1:], True
    return value, False


class ModelSchema(CamelCaseModel):
    """
    A serializable model architecture
    """

    @root_validator(pre=True)
    def check_types_defined(cls, values):
        layers = values.get("layers")
        featurizers = values.get("featurizers")
        layer_types = [layer.name for layer in generate.layers]
        featurizer_types = [featurizer.name for featurizer in generate.featurizers]
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
                    f"A featurizer has unknown type: {featurizer['type']}",
                    component_name=featurizer["name"],
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

        def _to_dict(arr: List[BaseModel]):
            return [item.dict() for item in arr]

        layers_and_featurizers = _to_dict(self.layers) + _to_dict(self.featurizers)
        for feat in layers_and_featurizers:
            for value in feat["constructor_args"].values():
                assert isinstance(
                    value, str
                ), "constructor_args values should be strings"
                reference, is_reference = _unwrap_dollar(value)
                if not is_reference and reference != "inputs":
                    continue
                reference = reference.split(".")[0]
                g.add_edge(feat["name"], reference)

        return g

    @classmethod
    def from_yaml(cls, yamlstr):
        config_dict = yaml.safe_load(yamlstr)
        return ModelSchema.parse_obj(config_dict)


if __name__ == "__main__":
    print(ModelSchema.schema_json())
