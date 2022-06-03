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


def access_attributes_of_interest(cls):

    module_name = get_module_name(cls)
    d = {}
    # TODO: Get the arguments types needed to instantiate the callable. That is, check if the __init__ code has some __code__ attr
    # If __init__ has no __code__, the callable has no instantiation args
    # This means there is an implementation for the constructorselfselfself.
    # args =  ___.__init__.__code__.co_varnames[:___.__init__.__code__.co_argscount]
    # defaults = ___.__init__.__defaults__
    # positionals = args[:len(defaults)]


    code = f'''
import {module_name}
if '__init__' in dir({cls}):
    if '__annotations__' in dir({cls}.__init__):
        d["types"] = {cls}.__init__.__annotations__
        d["defaults"] = {cls}.__init__.__defaults__
        d["call_type"] = "class"
    elif '__annotations__' in dir({cls}.__call__):
        d["types"] = {cls}.__call__.__annotations__
        d["defaults"] = {cls}.__call__.__defaults__
        d["call_type"] = "func"
if '__code__' in dir({cls}.__init__):
    d["args"] = {cls}.__init__.__code__.co_varnames
'''
    exec(code, globals(), { 'd': d })
    return d

def collect_components_info(class_paths: List[str]) -> Any:
    modules_memo = {}
    imported = {}
    for cls in class_paths:
        try:
            dic = access_attributes_of_interest(cls)
            print(dic)
            modules_memo[cls] = dic
        except Exception as exp:
            print(f'Failed for module {cls}: {str(exp)}')
    return modules_memo

memo = collect_components_info( torch_layers + torch_geometric_layers )
print(memo)

