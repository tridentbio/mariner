from typing import Any, List
from jinja2 import Environment, PackageLoader, select_autoescape

torch_layers = [
    'torch.nn.Linear',
    'torch.nn.Flatten',
]

torch_geometric_layers = [
    'torch_geometric.nn.GINConv',
    'torch_geometric.nn.global_add_pool',
]

python_primitives_reprs = {
    "<class 'int'>": "int",
    "<class 'float'>": "float",
    "<class 'bool'>": "bool",
    "<class 'string'>": "string",
}

def generate(prefix: str, path: str, arg_types: Any) -> str:
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

def get_module_name(classpath: str) -> str:
    return '.'.join(classpath.split('.')[:-1])


def access_attributes_of_interest(clspath):

    module_name = get_module_name(clspath)
    code = f'''
import {module_name}
references['cls'] = {clspath}
'''
    references = {} # cls must be a reference
    exec(code, globals(), { 'references': references })
    cls = references['cls']

    d = {}
    if '__init__' in dir(cls):
        if '__code__' in dir(cls.__init__):
            argscount = cls.__init__.__code__.co_argcount
            args = cls.__init__.__code__.co_varnames[:argscount] # Exclude kword args if any
            defaults = cls.__init__.__defaults__
            d["types"] = cls.__init__.__annotations__
            d["defaults"] = defaults
            d["call_type"] = "class"
            d["args"] = cls.__init__.__code__.co_varnames
        elif '__annotations__' in dir(cls):
            d["types"] = cls.__annotations__
            d["defaults"] = cls.__defaults__
            d["call_type"] = "func"
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

memo = collect_components_info( torch_layers + torch_geometric_layers )
print(memo)

