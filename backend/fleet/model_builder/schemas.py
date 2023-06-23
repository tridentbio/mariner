"""
Object schemas used by the model builder
"""
# Temporary file to hold all extracted mariner schemas
from typing import Any, Dict, List, Literal, Optional, Union

import networkx as nx
from pydantic import BaseModel, root_validator

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


class TorchModelSchema(CamelCaseModel, YAML_Model):
    """
    A serializable neural net architecture.
    """

    layers: List[LayersType] = []

    @root_validator(pre=True)
    def check_types_defined(cls, values):
        """Pydantic validator that checks if layers and featurizer types are
        known and secure (it's from one of the trusted 3rd party ML libs)

        Args:
            values (dict): dictionary with object values

        Raises:
            UnknownComponentType: in case a layer of featurizer has unknown ``type``
        """
        layers: List[LayersType] = values.get("layers")
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
        layers: List[LayersType] = values.get("layers")
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
            raise errors[0]
        return values

    def make_graph(self):
        """Makes a graph of the layers and featurizers

        The graph is used for a topological walk on the schema
        """
        model_graph = nx.DiGraph()
        for layer in self.layers:
            model_graph.add_node(layer.name)

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
                    model_graph.add_edge(reference, feat["name"], attr=key)
                elif isinstance(value, list):
                    for reference in value:
                        reference = reference.split(".")[0]
                        model_graph.add_edge(reference, feat["name"], attr=key)

                else:
                    continue
        return model_graph


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
