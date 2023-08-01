"""
This package is responsible to collect and get the options for building machine
learning models.
"""


import enum
from inspect import Parameter, signature
from typing import Any, Callable, Dict, List, Union, get_type_hints

import yaml
from humps import camel
from pydantic import BaseModel

from fleet.model_builder import generate
from fleet.model_builder.utils import get_class_from_path_string


def _get_documentation_link(class_path: str) -> Union[str, None]:
    """Create documentation link for the class_paths of pytorch
    and pygnn objects"""

    def is_from_pygnn(class_path: str) -> bool:
        return class_path.startswith("torch_geometric.")

    def is_from_pytorch(class_path: str) -> bool:
        return class_path.startswith("torch.")

    if is_from_pygnn(class_path):
        return (
            "https://pytorch-geometric.readthedocs.io/en/latest/modules/"
            f"nn.html#{class_path}"
        )
    elif is_from_pytorch(class_path):
        return (
            "https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#"
            f"{class_path}"
        )
    return None


def _get_docs(class_path: str) -> Union[str, None]:
    """Gets the documentation html string for the class_path"""
    try:
        docs = generate.sphinxfy(class_path)
        return docs
    except generate.EmptySphinxException:
        return None


def _get_return_type(cls: type, method_names: list[str]) -> Union[type, None]:
    """
    Gets the return type of the method in the class_path.

    Returns:
        The first return type found in the method_names.
    """
    for method in method_names:
        if method in dir(cls):
            forward_type_hints = get_type_hints(getattr(cls, method))
            output_type_hint = forward_type_hints.pop("return", None)
            return output_type_hint
    return None


def _get_default_constructor_args(
    cls: type, constructor_args_attribute: str
) -> Union[Dict[str, Any], None]:
    """
    Gets the default constructor args for the class_path.

    Returns:
        The default constructor args.
    """
    ctr_parameters = signature(
        signature(cls).parameters[constructor_args_attribute].annotation
    ).parameters
    return {
        name: parameter.default
        for name, parameter in ctr_parameters.items()
        if parameter.default is not parameter.empty
    }


class ComponentType(enum.Enum):
    """
    Defines the role of a component.
    """

    TRANSFORMER = "transformer"
    FEATURIZER = "featurizer"
    LAYER = "layer"
    SCIKIT_REG = "scikit_reg"
    SCIKIT_CLASS = "scikit_class"


class ArgumentOptionMetadata(BaseModel):
    """
    Models the metadata of an argument.

    Attributes:
        key: The name of the argument.
        label: The label of the argument.
        latex: The latex representation of the argument.
    """

    key: str
    label: Union[None, str]
    latex: Union[None, str]


ArgsOptions = Dict[str, List[Union[str, ArgumentOptionMetadata]]]


class Overrides(BaseModel):
    """
    Defines the model of the component_overrides.yml file.

    Attributes:
        args_options: possibly narrow string types to a subset of possible strings.
        defaults: override default value of an argument.
    """

    args_options: Union[ArgsOptions, None] = None
    defaults: Union[Dict[str, Any], None] = None


OptionOverrides = Dict[str, Overrides]


def _get_option_overrides() -> OptionOverrides:
    annotation_path = "fleet/model_builder/component_overrides.yml"
    with open(annotation_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
        return {key: Overrides(**value) for key, value in data.items()}


def is_bad(parameter: Parameter) -> bool:
    """Returns True if the parameter is bad

    Bad parameters are those that are not positional or keyword only

    Args:
        parameter (Parameter): the parameter

    Returns:
        bool: True if the parameter is bad else False
    """
    return parameter.annotation == Parameter.empty and type(
        parameter.default
    ) not in [int, str, float, bool]


def is_positional(parameter: Parameter) -> bool:
    """Returns True if the parameter is positional

    Args:
        parameter (Parameter): the parameter

    Returns:
        bool: True if the parameter is positional else False
    """
    return parameter.name != "self" and (
        (
            parameter.kind == Parameter.POSITIONAL_OR_KEYWORD
            and parameter.default == Parameter.empty
        )
        or parameter.kind == Parameter.POSITIONAL_ONLY
    )


def is_optional(parameter: Parameter) -> bool:
    """Returns True if the parameter is optional

    Args:
        parameter (Parameter): the parameter

    Returns:
        bool: True if the parameter is optional else False
    """
    return parameter.name != "self" and (
        parameter.kind == Parameter.KEYWORD_ONLY
        or (
            parameter.kind == Parameter.POSITIONAL_OR_KEYWORD
            and parameter.default != Parameter.empty
        )
    )


def _get_parameters(parameters: dict[str, Parameter]) -> dict[str, str]:
    parsed_parameters = {}
    for key, parameter in parameters.items():
        if is_bad(parameter):
            continue
        t = (
            parameter.annotation
            if parameter.annotation != Parameter.empty
            else type(parameter.default)
        )
        if is_optional(parameter):
            parsed_parameters[key] = f"{str(t)}?"
        elif is_positional(parameter):
            parsed_parameters[key] = str(t)

    return parsed_parameters


def _get_component(class_path: str) -> Any:
    cls = get_class_from_path_string(class_path)
    print(cls)
    constructor_args = _get_parameters(signature(cls).parameters)
    print(constructor_args)
    return {
        "constructorArgsSummary": constructor_args,
    }


class ComponentOption(BaseModel):
    """
    Models an option to build torch, sklearn or preprocessing pipelines.

    Attributes:
        class_path: The python class path from which the object can be
            imported.
        type: The role of the object.
        docs_link: A cross reference to the original documentation.
        docs: The html string of the documentation provided by help().
        output_type: A string that tells the output type of the object.
        default_args: A dictionary with the default arguments used to
            instantiate the object.
    """

    class_path: str
    component: Any
    type: ComponentType
    args_options: Union[ArgsOptions, None] = None
    docs_link: Union[str, None] = None
    docs: Union[str, None] = None
    output_type: Union[str, None] = None
    default_args: Union[dict, None] = None

    @classmethod
    def build(
        cls,
        class_path: str,
        type_: ComponentType,
        config_cls: type,
        overrides: Union[Overrides, None] = None,
        component: Union[Any, None] = None,
    ):
        """
        Makes a component option from the class_path.

        Args:
            class_path: The class_path of the component.

        Returns:
            A ComponentOption object.
        """
        docs_link = _get_documentation_link(class_path)
        # Todo: find better name for get_component
        component = component if component else _get_component(class_path)
        docs = _get_docs(class_path)
        described_cls = get_class_from_path_string(class_path)
        output_type = _get_return_type(described_cls, ["forward", "__call__"])
        try:
            default_args = (
                _get_default_constructor_args(config_cls, "constructorArgs")
                or {}
            )
            if overrides and overrides.defaults:
                for arg, default in overrides.defaults.items():
                    default_args[arg] = default
        except KeyError:
            default_args = None

        return ComponentOption(
            class_path=class_path,
            type=type_,
            docs_link=docs_link,
            docs=docs,
            output_type=str(output_type) if output_type else None,
            default_args=default_args,
            component=component,
            args_options=overrides.args_options if overrides else None,
        )

    class Config:  # pylint: disable=C0115
        alias_generator = camel.case
        allow_population_by_field_name = True
        allow_population_by_alias = True
        underscore_attrs_are_private = True


class ComponentOptionsManager:
    """
    Manages the options for building machine learning models.
    """

    def __init__(self) -> None:
        self.options: List[ComponentOption] = []
        self.overrides = _get_option_overrides()

    def configoption(
        self,
        component_type: ComponentType,
        class_path_attribute="type",
        summary_cls: Union[None, Callable] = None,
    ):
        """
        Decorator to register a class path as a component option.
        """

        def decorator(cls):
            class_path = (
                signature(cls).parameters[class_path_attribute].default
            )
            self.options.append(
                ComponentOption.build(
                    class_path,
                    component_type,
                    cls,
                    overrides=self.overrides.get(class_path),
                    component=summary_cls() if summary_cls else None,
                )
            )
            return cls

        return decorator

    def config_transformer(
        self,
        **kwargs,
    ):
        """
        Decorator to register a class path as a transformer component option.
        """
        return self.configoption(ComponentType.TRANSFORMER, "type", **kwargs)

    def config_featurizer(self, **kwargs):
        """
        Decorator to register a class path as a featurizer component option.
        """
        return self.configoption(ComponentType.FEATURIZER, "type", **kwargs)

    def config_layer(self, **kwargs):
        """
        Decorator to register a class path as a layer component option.
        """
        return self.configoption(ComponentType.LAYER, "type", **kwargs)

    def config_scikit_reg(self, **kwargs):
        """
        Decorator to register a class path as a scikit-learn regressor component option.
        """
        return self.configoption(ComponentType.SCIKIT_REG, "type", **kwargs)

    def config_scikit_class(self, **kwargs):
        """
        Decorator to register a class path as a scikit-learn classifier component option.
        """
        return self.configoption(ComponentType.SCIKIT_CLASS, "type", **kwargs)

    def import_libs(
        self,
    ):
        """
        Load all the scripts that are needed for populating options_manager.
        """
        # import scripts that are needed for populating options_manager.
        import fleet.model_builder.layers_schema  # pylint: disable=C0415,W0611
        import fleet.preprocessing  # pylint: disable=C0415,W0611
        import fleet.scikit_.schemas  # pylint: disable=C0415,W0611


options_manager = ComponentOptionsManager()