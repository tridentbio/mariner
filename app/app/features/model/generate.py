from typing import Any, List
from jinja2 import Environment, PackageLoader, select_autoescape
from humps import camel

torch_layers = [
    'torch.nn.Linear',
    'torch.nn.Flatten',
]

torch_geometric_layers = [
    'torch_geometric.nn.GINConv',
    'torch_geometric.nn.global_add_pool',
]

python_primitives_reprs = {
    int: "int",
    float: "float",
    bool: "bool",
    str: "string",
}


def get_module_name(classpath: str) -> str:
    return '.'.join(classpath.split('.')[:-1])

def get_class_from_path_string(pathstring: str):
    module_name = get_module_name(pathstring)
    code = f'''
import {module_name}
references['cls'] = {pathstring}
'''
    references = {} # cls must be a reference
    exec(code, globals(), { 'references': references })
    return references['cls']

def access_attributes_of_interest(clspath):
    cls = get_class_from_path_string(clspath)
    d = {}
    if '__init__' in dir(cls):
        if '__code__' in dir(cls.__init__):
            argscount = cls.__init__.__code__.co_argcount
            args = cls.__init__.__code__.co_varnames[:argscount] # Exclude kword args if any
            defaults = cls.__init__.__defaults__
            d["types"] = cls.__init__.__annotations__ # dictionary mapping argname to type
            d["defaults"] = defaults # tuple with ending default argument  valuts
            d["call_type"] = "class"
            d["args"] = args # tuple with argnames.
            d["not_defaults"] = args[1:argscount-len(defaults)]
        elif '__annotations__' in dir(cls):
            d["types"] = cls.__annotations__
            d["defaults"] = cls.__defaults__
            d["call_type"] = "func"
            argscount = cls.__code__.co_argcount
            args = cls.__code__.co_varnames[:argscount] # Exclude kword args if any

            d["not_defaults"] = args[1:argscount-len(d['defaults'])]
            d["args"] = args
    return d

def collect_components_info(class_paths: List[str]) -> Any:
    modules_memo = {}
    for cls in class_paths:
        try:
            dic = access_attributes_of_interest(cls)
            modules_memo[cls] = dic
        except Exception as exp:
            print(f'Failed for module {cls}: {str(exp)}')
    return modules_memo


def is_primitive(pr):
    return pr == int or pr == str or pr == bool


def get_component_template_args(path: str):
    path_parts = path.split('.')
    if '_' in path_parts[-1]:
        compname = camel.case(path_parts[-1])
    else:
        compname = path_parts[-1]
    prefix = camel.case(path_parts[0]) + compname
    prefix = prefix.title()
    info = collect_components_info([path])[path]
    arg_types = {
        argname: python_primitives_reprs[info['types'][argname]]
        for argname in info['not_defaults']
        if argname in info['types']
        and is_primitive(info['types'][argname])
    }
    return prefix, arg_types

def generate(path: str) -> str:
    prefix, arg_types = get_component_template_args(path)
    env = Environment(
        loader=PackageLoader("app.features.model"),
        autoescape=select_autoescape()
    )
    template = env.get_template('component.py.j2')
    return template.render(
        prefix=prefix,
        path=path,
        arg_types=arg_types
    )

def generate_bundle(paths: List[str]) -> str:
    env = Environment(
        loader=PackageLoader("app.features.model"),
        autoescape=select_autoescape()
    )
    template = env.get_template('base.py.jinja')
    args = [ get_component_template_args(path) for path in paths ]
    components = [
        { "prefix": prefix, "arg_types": arg_types, "path": path }
        for (prefix, arg_types), path in zip(args, paths)
    ]
    return template.render(
        components=components
    )
    

if __name__ == '__main__':
    import sys
    template = sys.argv[1]
    if template == 'component':
        compnames = sys.argv[2:]
        for compname in compnames:
            print(generate(compname))
    elif template == 'base':
        print(generate_bundle(torch_layers + torch_geometric_layers))


