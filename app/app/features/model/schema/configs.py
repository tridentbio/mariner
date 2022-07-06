from ast import literal_eval
from typing import List, Literal, Optional, Union

import networkx as nx
import yaml

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


class ModelOptions(ApiBaseModel):
    layers: List[LayersArgsType]
    featurizers: List[FeaturizersArgsType]
    component_annotations: List[LayerAnnotation]


class ModelConfig(ApiBaseModel):
    name: str
    dataset: DatasetConfig
    featurizers: List[FeaturizersType]
    layers: List[LayersType]

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
