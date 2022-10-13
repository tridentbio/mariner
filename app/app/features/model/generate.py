from typing import Optional, get_type_hints
from dataclasses import dataclass
from typing import Any, List
import typing

from humps import camel
from jinja2 import Environment, PackageLoader, select_autoescape

from app.features.model.utils import get_class_from_path_string


@dataclass
class Layer:
    name: str


featurizers = [
    Layer(name) for name in ["app.features.model.featurizers.MoleculeFeaturizer"]
]

layers = [
    Layer(name)
    for name in [
        "app.features.model.layers.OneHot",
        "app.features.model.layers.GlobalPooling",
        "app.features.model.layers.Concat",
        "torch.nn.Linear",
        "torch.nn.Sigmoid",
        "torch.nn.ReLU",
        "torch_geometric.nn.GCNConv",
    ]
]


python_primitives_reprs = {
    int: "int",
    float: "float",
    bool: "bool",
    str: "string",
}


def get_module_name(classpath: str) -> str:
    return ".".join(classpath.split(".")[:-1])


# deprecated
def access_attributes_of_interest(clspath):
    cls = get_class_from_path_string(clspath)
    if "__init__" in dir(cls) and "__code__" in dir(cls.__init__):
        d = {}
        argscount = cls.__init__.__code__.co_argcount
        args = cls.__init__.__code__.co_varnames[
            :argscount
        ]  # Exclude kword args if any
        defaults = cls.__init__.__defaults__
        d["types"] = cls.__init__.__annotations__  # dictionary mapping argname to type
        d["defaults"] = defaults  # tuple with ending default argument  valuts
        d["call_type"] = "class"
        d["args"] = args  # tuple with argnames.
        last = argscount - len(defaults or [])
        d["not_defaults"] = args[1:last] if defaults else args[1:]
        return d
    raise Exception(
        f"Failed to inspect type annotations for {clspath}. "
        "Is it a class with __init__ implementation?"
    )


# deprecated
def collect_components_info(class_paths: List[str]) -> Any:
    modules_memo = {}
    for cls in class_paths:
        try:
            dic = access_attributes_of_interest(cls)
            modules_memo[cls] = dic
        except Exception as exp:
            print(f"Failed for module {cls}: {str(exp)}")
            raise exp
    return modules_memo


def is_primitive(pr):
    return pr in [int, str, bool, float]


# deprecated
def get_component_template_args(path: str):
    path_parts = path.split(".")
    if "_" in path_parts[-1]:
        compname = camel.case(path_parts[-1])
    else:
        compname = path_parts[-1]
    prefix = camel.case(path_parts[0]) + compname
    prefix = prefix.title()
    info = collect_components_info([path])[path]
    arg_types = {
        argname: python_primitives_reprs[info["types"][argname]]
        for argname in info["not_defaults"]
        if argname in info["types"] and is_primitive(info["types"][argname])
    }
    return prefix, arg_types


class Argument:
    """
    Represents an argument in a function signature

    Attributes:
        required:
        name:
        type:

    """

    def __init__(self, required: bool, name: str, type: typing.Any):
        self.required = required
        self.name = name
        self.type = type


def args_to_list(args: List[Argument]) -> List[tuple[str, Any]]:
    """
    Maps a list of arguments back to a list of (arg_name, arg_type) tuple

    Args:
        args (List[Argument]):

    Returns:
        List[tuple[str, Any]]


    """
    return [(arg.name, arg.type) for arg in args]


class Signature:
    """
    Represents the signature of a function call

    Attributes:
        args(List[Argument]): A summary of the arguments involved in the function call
        return_type(Any): The return type of the function
    """

    def __init__(
        self,
        pos_args_names_and_types: List[tuple[str, Any]],
        kwargs_names_and_types: List[tuple[str, Any]],
        return_type: Any,
    ):
        self.args = [
            Argument(name=name, type=type, required=True)
            for name, type in pos_args_names_and_types
        ] + [
            Argument(name=name, type=type, required=False)
            for name, type in kwargs_names_and_types
        ]
        self.return_type = return_type

    def get_positional_arguments(self) -> List[Argument]:
        return [arg for arg in self.args if arg.required]

    def get_keyword_arguments(self) -> List[Argument]:
        return [arg for arg in self.args if not arg.required]


def get_component_constructor_signature(
    class_def: Any,
) -> Optional[Signature]:
    """
    Get's  the constructor Signature

    Args:
        class_def (Any):

    Returns:
        Signature

    """
    return _get_component_signature(class_def, "__init__")


def get_component_forward_signature(class_def: Any) -> Optional[Signature]:
    """
    Get's the forward ( or __call__ for featurizers ) Signature

    Args:
        class_def (Any):

    Returns:
        Signature
    """
    if "forward" in dir(class_def):
        return _get_component_signature(class_def, "forward")
    elif "__call__" in dir(class_def):
        return _get_component_signature(class_def, "__call__")
    return None


def _get_component_signature(class_def: Any, method_name: str) -> Optional[Signature]:
    if method_name not in dir(class_def):
        return None
    method = getattr(class_def, method_name)
    if "__code__" not in dir(method):
        raise Exception(
            "Unsure how to introspect method {method_name} with no __code__"
        )
    positional_argscount = class_def.__init__.__code__.co_argcount
    positional_arg_names = class_def.__init__.__code__.co_varnames[
        :positional_argscount
    ]
    type_hints = get_type_hints(method)
    args_names_and_types = [
        (arg_name, arg_type)
        for arg_name, arg_type in type_hints.items()
        if arg_name in positional_arg_names
    ]
    kwargs_names_and_types = [
        (arg_name, arg_type)
        for arg_name, arg_type in type_hints.items()
        if arg_name not in positional_arg_names and arg_name != "return"
    ]
    return Signature(
        pos_args_names_and_types=args_names_and_types,
        kwargs_names_and_types=kwargs_names_and_types,
        return_type=type_hints.get("return", None),
    )


def get_component_template_args_v2(path: str):
    """
    Get's the dictionary with the arguments passed to jinja template engine to render the autogenerated
    code for a component (layer or featurizer)

    Below is a description of the dictionary built:
        - "prefix"
        - "path"
        - "ctr": A description of the constructor signature of the class referenced by path
            - "pos": The positional (required) arguments. A list of tuple[str, Any] composed of (arg_name, arg_type)
            - "kw": The keyword (presumably optional) arguments. Same as pos.
        - "fwd": The same as ctr, but for the forward. If class referenced does not have a forward, we look the __call__ method instead
            - "pos":
            - "kw":
    """
    path_parts = path.split(".")
    if "_" in path_parts[-1]:
        compname = camel.case(path_parts[-1])
    else:
        compname = path_parts[-1]
    prefix = camel.case(path_parts[0]) + compname
    prefix = prefix.title()
    class_def = get_class_from_path_string(path)
    ctr_signature = get_component_constructor_signature(class_def)
    forward_signature = get_component_forward_signature(class_def)
    if ctr_signature:
        ctr = {
            "pos": args_to_list(ctr_signature.get_positional_arguments()),
            "kw": args_to_list(ctr_signature.get_keyword_arguments()),
        }
    else:
        ctr = {}
    if forward_signature:
        fwd = {
            "pos": args_to_list(forward_signature.get_positional_arguments()),
            "kw": args_to_list(forward_signature.get_keyword_arguments()),
        }
    else:
        fwd = {}

    return {"prefix": prefix, "path": path, "ctr": ctr, "fwd": fwd}


def create_jinja_env():
    env = Environment(
        loader=PackageLoader("app.features.model"), autoescape=select_autoescape()
    )

    def type_name(value):
        if value == "string":
            return "str"
        return value

    env.globals.update(type_name=type_name)
    return env


def generate(path: str) -> str:
    prefix, arg_types = get_component_template_args(path)
    env = create_jinja_env()
    template = env.get_template("component.py.jinja")
    return template.render(
        component={"prefix": prefix, "path": path, "arg_types": arg_types}
    )


def generatev2(path: str) -> str:
    template_args = get_component_template_args_v2(path)
    env = create_jinja_env()
    template = env.get_template("componentv2.py.jinja")
    return template.render(**template_args)


def generate_bundle() -> str:
    env = create_jinja_env()
    schemas_template = env.get_template("base.py.jinja")
    args = [get_component_template_args(layer.name) for layer in layers]
    layer_components = [
        {
            "prefix": prefix,
            "arg_types": arg_types,
            "path": layer.name,
        }
        for (prefix, arg_types), layer in zip(args, layers)
    ]
    args = [get_component_template_args(featurizer.name) for featurizer in featurizers]
    featurizer_components = [
        {
            "prefix": prefix,
            "arg_types": arg_types,
            "path": layer.name,
        }
        for (prefix, arg_types), layer in zip(args, featurizers)
    ]

    bundled_schema = schemas_template.render(
        layer_components=layer_components, featurizer_components=featurizer_components
    )
    return bundled_schema


def generate_bundlev2() -> str:
    env = create_jinja_env()
    schemas_template = env.get_template("base.py.jinja")
    layer_template_args = [
        get_component_template_args_v2(layer.name) for layer in layers
    ]
    featurizer_template_args = [
        get_component_template_args_v2(featurizer.name) for featurizer in featurizers
    ]
    return schemas_template.render(
        layer_components=layer_template_args,
        featurizer_components=featurizer_template_args,
    )


if __name__ == "__main__":
    import sys

    template = sys.argv[1]
    if template == "component":
        compnames = sys.argv[2:]
        for compname in compnames:
            print(generate(compname))
    elif template == "base":
        bundle = generate_bundle()
        print(bundle)
        sys.exit(0)
