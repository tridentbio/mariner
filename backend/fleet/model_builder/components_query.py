"""
Functions to query ModelSchema
"""
from typing import Optional, get_args, get_type_hints

from pydantic import BaseModel

from fleet.model_builder.layers_schema import FeaturizersType, LayersType


def get_component_config_by_type(name: str) -> Optional[BaseModel]:
    """Gets a layer or featurizer class by type (.e.g "torch.nn.Linear"

    Args:
        name: type, .e.g "torch.nn.Linear"

    Returns:
        the Config class for the given type
    """
    # lazy import is necessary to avoid a tricky dependency loop during
    # code generation
    layer_types = [component for component in get_args(LayersType)]
    featurizer_types = [component for component in get_args(FeaturizersType)]
    for component in layer_types + featurizer_types:
        if component.construct().type == name:
            return component


def get_component_constructor_args_by_type(name: str) -> Optional[BaseModel]:
    """Gets the ConstructorArgs class for a component by type

    Args:
        name: type, .e.g "torch.nn.Linear"

    Returns:
        The ConstructorArgs for the given type
    """

    component = get_component_config_by_type(name)
    if component:
        type_hints = get_type_hints(component)
        if "constructor_args" in type_hints:
            return type_hints["constructor_args"]
