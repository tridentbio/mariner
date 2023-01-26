"""
This package defines useful queries on a ModelSchema. When interacting with the model
schema, developers should prefer using this API over accessing attributes directly from
the schema in order to maintain a single point of change (as best as possible) on schema
updates.
"""
from typing import Any, Callable, List, Literal, Set, Union

from model_builder.layers_schema import FeaturizersType, LayersType
from model_builder.schemas import ColumnConfig, ModelSchema
from model_builder.utils import unwrap_dollar


def get_columns_configs(config: ModelSchema) -> List[ColumnConfig]:
    """Get's the column configs from targets and featurizers of a ModelSchema object

    Args:
        config: ModelSchema

    Returns:
        List[ColumnConfig]
    """
    return config.dataset.feature_columns + [config.dataset.target_column]


def get_column_config(
    config: ModelSchema, column_name: str
) -> Union[ColumnConfig, None]:
    """Get's the column config of any column (target or featurizer) with name
    equals ``column_name``
    Args:
        config: ModelSchem
        column_name: str

    Returns:
        ColumnConfig
    """
    columns = get_columns_configs(config)
    for column in columns:
        if column.name == column_name:
            return column
    return None


def get_dependencies(component_config: Union[LayersType, FeaturizersType]) -> Set[str]:
    """Get a layer or featurizer dependencies

    Returns set of node names that are dependencies of ``component_config`` by looking
    at it's ``forward_args `` attribute'

    Args:
        component_config: a config for a layer or featurizer

    Returns:
        Set[str]: a set of node names on which ``component_config`` depends
    """
    deps = set()
    for input_str in component_config.forward_args.dict().values():
        if isinstance(input_str, str):
            dep, isref = unwrap_dollar(input_str)
            if not isref:
                continue
            deps.add(dep)
        elif isinstance(input_str, list):
            for item in input_str:
                dep, isref = unwrap_dollar(item)
                if not isref:
                    continue
                deps.add(dep)
    return deps


def get_target_columns(model_schema: ModelSchema) -> List[ColumnConfig]:
    """Get's the target columns in the model_schema

    Args:
        model_schema: model config

    Returns:
        List[ColumnConfig]: list of columns
    """
    return [model_schema.dataset.target_column]


Node = Union[LayersType, FeaturizersType, ColumnConfig]
NodeType = Literal["layer", "featurizer", "input", "output"]


def iterate_schema(config: ModelSchema, callback: Callable[[Node, NodeType], Any]):
    """Walks the schema

    Iterates the inputs, featurizers, layers and output nodes in the computational graph,
    calling ``callback`` to each.

    Args:
        config: the model schema to iterate
        callback: the function to be executed on the node


    Examples:
        >>> config = ModelSchema(...)
        >>> def on_node(node: Node, type: NodeType):
                print("at node %r of type %s" % (node, type))
        >>> iterate_schema(config, on_node)

    """
    ...
