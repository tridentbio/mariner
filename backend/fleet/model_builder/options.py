"""
Model builder options. Most options are get through previous code
inspection during code generation. Options may be further updated
using the `component_overrides.yml` file.
"""
import functools
from inspect import Parameter, signature
from typing import Any, Dict, List, Literal, Optional, Union, get_type_hints

import yaml
from pydantic import BaseModel

from fleet.model_builder import generate, layers_schema
from fleet.model_builder.schemas import ComponentOption
from fleet.model_builder.utils import get_class_from_path_string


class Overrides(BaseModel):
    """
    Defines the model of the component_overrides.yml file.

    Attributes:
        args_options: possibly narrow string types to a subset of possible strings.
        defaults: override default value of an argument.
    """

    args_options: Optional[Dict[str, Any]]
    defaults: Optional[Dict[str, Any]]


OptionOverrides = Dict[str, Overrides]


def _get_option_overrides() -> OptionOverrides:
    annotation_path = "fleet/model_builder/component_overrides.yml"
    with open(annotation_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
        return {key: Overrides(**value) for key, value in data.items()}


def _get_documentation_link(class_path: str) -> Optional[str]:
    """Create documentation link for the class_paths of pytorch
    and pygnn objects"""

    def is_from_pygnn(class_path: str) -> bool:
        return class_path.startswith("torch_geometric.")

    def is_from_pytorch(class_path: str) -> bool:
        return class_path.startswith("torch.")

    if is_from_pygnn(class_path):
        return f"https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#{class_path}"  # noqa: E501
    elif is_from_pytorch(class_path):
        return f"https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#{class_path}"  # noqa: E501
    return None


def _get_annotations_from_cls(cls_path: str) -> ComponentOption:
    """Gives metadata information of the component implemented by `cls_path`"""
    docs_link = _get_documentation_link(cls_path)
    cls = get_class_from_path_string(cls_path)
    try:
        docs = generate.sphinxfy(cls_path)
    except generate.EmptySphinxException:
        docs = cls_path
    forward_type_hints = {}
    if "forward" in dir(cls):
        forward_type_hints = get_type_hints(getattr(cls, "forward"))
    elif "__call__" in dir(cls):
        forward_type_hints = get_type_hints(getattr(cls, "__call__"))
    output_type_hint = forward_type_hints.pop("return", None)
    return ComponentOption.construct(
        docs_link=docs_link,
        docs=docs,
        output_type=str(output_type_hint) if output_type_hint else None,
        class_path=cls_path,
        type=None,  # type: ignore
        component=None,  # type: ignore
        default_args=None,  # type: ignore
    )


@functools.cache
def get_model_options() -> List[ComponentOption]:
    """Gets all component (featurizers and layer) options supported by the system,
    along with metadata about each"""
    layer_types = [layer.name for layer in generate.layers]
    featurizer_types = [f.name for f in generate.featurizers]

    def get_default_values(summary_name: str) -> Union[Dict[str, Any], None]:
        try:
            class_args = getattr(
                layers_schema,
                summary_name.replace("Summary", "ConstructorArgs"),
            )
            args = {
                arg.name: arg.default
                for arg in signature(class_args).parameters.values()
                if arg.default != Parameter.empty
            }
            return args
        except AttributeError:
            return None

    def get_args_options(classpath: str) -> Union[Dict[str, List[str]], None]:
        overrides = _get_option_overrides()
        return (
            overrides[classpath].args_options
            if classpath in overrides
            else None
        )

    def get_summary_and_constructor_args(
        cls_path: str,
    ):
        for schema_exported in dir(layers_schema):
            if (
                schema_exported.endswith("Summary")
                and not schema_exported.endswith("ForwardArgsSummary")
                and not schema_exported.endswith("ConstructorArgsSummary")
            ):
                class_def = getattr(layers_schema, schema_exported)
                default_args = get_default_values(schema_exported)
                instance = class_def()
                if instance.type and instance.type == cls_path:
                    return instance, default_args
        raise RuntimeError(f"Schema for {cls_path} not found")

    component_annotations: List[ComponentOption] = []

    overrides = _get_option_overrides()

    def make_component(class_path: str, type_: Literal["layer", "featurizer"]):
        summary = get_summary_and_constructor_args(class_path)
        if not summary:
            return None
        summary, default_args = summary
        option = _get_annotations_from_cls(class_path)
        option.args_options = get_args_options(class_path)
        option.type = type_
        option.component = summary
        option.default_args = default_args or {}
        override = overrides[class_path] if class_path in overrides else None
        if override and override.defaults:
            for key, value in override.defaults.items():
                option.default_args[key] = value
        return option

    for class_path in layer_types:
        component = make_component(class_path, "layer")
        if component:
            component_annotations.append(component)

    for class_path in featurizer_types:
        component = make_component(class_path, "featurizer")
        if component:
            component_annotations.append(component)

    return component_annotations
