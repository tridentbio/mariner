from ast import literal_eval
from typing import List, Literal, Optional, Union, get_args, get_type_hints

import networkx as nx
import yaml
from pydantic import ValidationError, root_validator

from app.features.model.generate import get_component_args_by_type
from app.features.model.schema.layers_schema import (
    FeaturizersArgsType,
    FeaturizersType,
    LayersArgsType,
    LayersType,
)
from app.schemas.api import ApiBaseModel


class Tuple(str):
    val: tuple

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(examples=["(1,)", "(1,1,0)"])

    @classmethod
    def validate(cls, v):
        try:
            t = literal_eval(v)
            if not isinstance(t, tuple):
                raise ValueError("Tuple(s), s should evaluate to a tuple")
            return cls(v)
        except Exception:
            raise


class CycleInGraphException(ApiBaseModel):
    """
    Raised when cycles are detected in the computational
    graph of the model
    """

    code = 200
    message = "There is a cycle in the graph"


class DatasetConfig(ApiBaseModel):
    name: str
    target_column: str
    feature_columns: List[str]


MessagePassingRule = Literal["graph-receiver"]
InputsSameTypeRule = Literal["inputs-same-type"]

LayerRule = Union[MessagePassingRule, InputsSameTypeRule]


class LayerAnnotation(ApiBaseModel):
    docs_link: Optional[str]
    docs: Optional[str]
    num_inputs: int
    num_outputs: int
    rules: List[LayerRule]
    class_path: str


class ModelOptions(ApiBaseModel):
    layers: List[LayersArgsType]
    featurizers: List[FeaturizersArgsType]
    component_annotations: List[LayerAnnotation]


class UnknownComponentType(ValueError):
    component_name: str

    def __init__(self, *args, component_name: str):
        super().__init__(*args)
        self.component_name = component_name


class MissingComponentArgs(ValueError):
    component_name: str
    missing: List[Union[str, int]]

    def __init__(self, *args, missing: List[Union[str, int]], component_name: str):
        super().__init__(*args)
        self.component_name = component_name
        self.missing = missing


class ModelConfig(ApiBaseModel):
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
            print("gonna validate now")
            try:
                args_cls.validate(layer["args"])
            except ValidationError as exp:
                from pprint import pprint

                pprint(exp)
                errors += [
                    MissingComponentArgs(
                        missing=[l for l in error["loc"]], component_name=layer["name"]
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
                        missing=[l for l in error["loc"]],
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
        return ModelConfig.parse_obj(config_dict)


if __name__ == "__main__":
    print(ModelConfig.schema_json())
