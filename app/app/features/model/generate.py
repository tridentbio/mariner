from dataclasses import dataclass
from typing import Any, List

from humps import camel
from jinja2 import Environment, PackageLoader, select_autoescape

from app.features.model.utils import get_class_from_path_string


@dataclass
class Layer:
    name: str
    forward_input_mask: int = 0b1000


"""
    Layers that receive a tensor as inputs
    of the forward call
"""
single_input_layer_modules = [
    "torch.nn.Linear",
    "torch.nn.Flatten",
]
single_input_layers = [Layer(name, 0b1000) for name in single_input_layer_modules]

"""
    Layers that receive a tensor of node features
    and edge_index as inputs of the forward call
"""
edge_index_layer_modules = [
    "torch_geometric.nn.GINConv",
    "torch_geometric.nn.GCNConv",
]
edge_index_layers = [Layer(name, 0b1100) for name in edge_index_layer_modules]

"""
    Layers that receive a tensor of node features
    and a torch_geometric Batch as inputs of the 
    in the forward call
"""
batch_layer_modules = [
    "torch_geometric.nn.global_add_pool",
]
batch_layers = [Layer(name, 0b1001) for name in batch_layer_modules]

layers = single_input_layers + edge_index_layers + batch_layers

python_primitives_reprs = {
    int: "int",
    float: "float",
    bool: "bool",
    str: "string",
}


def get_module_name(classpath: str) -> str:
    return ".".join(classpath.split(".")[:-1])


def access_attributes_of_interest(clspath):
    cls = get_class_from_path_string(clspath)
    d = {}
    if "__init__" in dir(cls):
        if "__code__" in dir(cls.__init__):
            argscount = cls.__init__.__code__.co_argcount
            args = cls.__init__.__code__.co_varnames[
                :argscount
            ]  # Exclude kword args if any
            defaults = cls.__init__.__defaults__
            d[
                "types"
            ] = cls.__init__.__annotations__  # dictionary mapping argname to type
            d["defaults"] = defaults  # tuple with ending default argument  valuts
            d["call_type"] = "class"
            d["args"] = args  # tuple with argnames.
            d["not_defaults"] = args[1 : argscount - len(defaults)]
        elif "__annotations__" in dir(cls):
            d["types"] = cls.__annotations__
            d["defaults"] = cls.__defaults__
            d["call_type"] = "func"
            argscount = cls.__code__.co_argcount
            args = cls.__code__.co_varnames[:argscount]  # Exclude kword args if any

            d["not_defaults"] = args[1 : argscount - len(d["defaults"])]
            d["args"] = args
    return d


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
    return pr == int or pr == str or pr == bool or pr == float


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


def generate(path: str) -> str:
    prefix, arg_types = get_component_template_args(path)
    env = Environment(
        loader=PackageLoader("app.features.model"), autoescape=select_autoescape()
    )
    template = env.get_template("component.py.j2")
    return template.render(prefix=prefix, path=path, arg_types=arg_types)


def generate_bundle(layers: List[Layer]) -> str:
    env = Environment(
        loader=PackageLoader("app.features.model"), autoescape=select_autoescape()
    )
    schemas_template = env.get_template("base.py.jinja")
    args = [get_component_template_args(layer.name) for layer in layers]
    components = [
        {
            "prefix": prefix,
            "arg_types": arg_types,
            "path": layer.name,
            "forward_input_mask": layer.forward_input_mask,
        }
        for (prefix, arg_types), layer in zip(args, layers)
    ]
    bundled_schema = schemas_template.render(components=components)
    return bundled_schema


if __name__ == "__main__":
    import sys

    template = sys.argv[1]
    if template == "component":
        compnames = sys.argv[2:]
        for compname in compnames:
            print(generate(compname))
    elif template == "base":
        print(generate_bundle(layers))
