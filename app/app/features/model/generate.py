from dataclasses import dataclass
from typing import Any, List

from humps import camel
from jinja2 import Environment, PackageLoader, select_autoescape

from app.features.model.utils import get_class_from_path_string


@dataclass
class Layer:
    name: str


featurizers = [
    Layer(name) for name in [
        'app.features.model.featurizers.MoleculeFeaturizer'
    ]
]

layers = [
    Layer(name) for name in [
        'app.features.model.layers.GlobalPooling',
        'app.features.model.layers.Concat',
        'torch.nn.Linear',
        'torch.nn.Sigmoid',
        'torch.nn.ReLU',
        'torch_geometric.nn.GCNConv',
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


def access_attributes_of_interest(clspath):
    cls = get_class_from_path_string(clspath)
    if "__init__" in dir(cls) and '__code__' in dir(cls.__init__):
        d = {}
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
        d["not_defaults"] = args[1 : argscount - len(defaults)] if defaults else args[1:]
        return d
    raise Exception(f'Failed to inspect type annotations for {clspath}. Is it a class with __init__ implementation?')


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

def create_jinja_env():
    env = Environment(
        loader=PackageLoader("app.features.model"), autoescape=select_autoescape()
    )
    def type_name(value):
      if value == 'string':
        return 'str'
      return value
    env.globals.update(type_name=type_name)
    return env


def generate(path: str) -> str:
    prefix, arg_types = get_component_template_args(path)
    env = create_jinja_env()
    template = env.get_template("component.py.jinja")
    return template.render(component={
        'prefix':prefix, 'path': path, 'arg_types': arg_types
    })


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
        layer_components=layer_components,
        featurizer_components=featurizer_components
    )
    return bundled_schema


if __name__ == "__main__":
    import sys

    template = sys.argv[1]
    if template == "component":
        compnames = sys.argv[2:]
        for compname in compnames:
            print(generate(compname))
    elif template == "base":
        print(generate_bundle())
