from typing import Optional, get_args, get_type_hints

from pydantic import BaseModel

from model_builder.layers_schema import FeaturizersType, LayersType


def get_component_config_by_type(name: str) -> Optional[BaseModel]:
    layer_types = [c for c in get_args(LayersType)]
    featurizer_types = [FeaturizersType]
    for component in layer_types + featurizer_types:
        th = get_type_hints(component)
        if get_args(th["type"])[0] == name:
            return component


def get_component_args_by_type(name: str) -> Optional[BaseModel]:
    component = get_component_config_by_type(name)
    if component:
        th = get_type_hints(component)
        if "args" in th:
            return th["args"]
