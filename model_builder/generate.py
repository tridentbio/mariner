import functools
import inspect
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from inspect import Parameter, Signature, signature
from typing import Any, List, Optional

from humps import camel
from jinja2 import Environment, PackageLoader, select_autoescape
from sphinx.application import Sphinx

from model_builder.utils import get_class_from_path_string


@dataclass
class Layer:
    name: str


featurizers = [
    Layer(name)
    for name in [
        "model_builder.featurizers.MoleculeFeaturizer",
        # PyG's from_smiles only outputs Long tensors for node_features
        # Waiting for issue ... to be resolved
        # "model_builder.featurizers.FromSmiles",
    ]
]

layers = [
    Layer(name)
    for name in [
        "model_builder.layers.OneHot",
        "model_builder.layers.GlobalPooling",
        "model_builder.layers.Concat",
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
        d["not_defaults"] = args[1:last]
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
    return get_component_signature(class_def, "__init__")


def get_component_forward_signature(class_def: Any) -> Optional[Signature]:
    """
    Get's the forward ( or __call__ for featurizers ) Signature

    Args:
        class_def (Any):

    Returns:
        Signature
    """
    if "forward" in dir(class_def):
        return get_component_signature(class_def, "forward")
    elif "__call__" in dir(class_def):
        return get_component_signature(class_def, "__call__")
    return None


def get_component_signature(class_def: Any, method_name: str) -> Optional[Signature]:
    if method_name not in dir(class_def):
        return None
    method = getattr(class_def, method_name)
    if "__code__" not in dir(method):
        raise Exception(
            "Unsure how to introspect method {method_name} with no __code__"
        )
    return signature(method)


def is_bad(parameter: Parameter):
    return parameter.annotation == inspect._empty


def is_positional(parameter: Parameter):
    return parameter.name != "self" and (
        (
            parameter.kind == Parameter.POSITIONAL_OR_KEYWORD
            and parameter.default == Parameter.empty
        )
        or parameter.kind == Parameter.POSITIONAL_ONLY
    )


def is_optional(parameter: Parameter):
    return parameter.name != "self" and (
        parameter.kind == Parameter.KEYWORD_ONLY
        or (
            parameter.kind == Parameter.POSITIONAL_OR_KEYWORD
            and parameter.default != inspect._empty
        )
    )


def args_to_list(params: List[Parameter]) -> List[tuple[str, Any, Any]]:
    return [(param.name, str(param.annotation), param.default) for param in params]


def get_component_template_args_v2(path: str):
    """
    Get's the dictionary with the arguments passed to jinja template engine to render the autogenerated
    code for a component (layer or featurizer)

    Below is a description of the dictionary built:
        - "prefix"
        - "path"
        - "ctr": A description of the constructor signature of the class referenced by path
            - "pos": The positional (required) arguments. A list of tuple[str, Any, None] composed of (arg_name, arg_type, None)
            - "kw": The keyword (presumably optional) arguments. tuple[str, Any, Any] composed of (arg_name, arg_type, arg_default)
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
    fwd_signature = get_component_forward_signature(class_def)

    if ctr_signature:
        ctr = {
            "pos": args_to_list(
                [
                    param
                    for param in ctr_signature.parameters.values()
                    if is_positional(param) and not is_bad(param)
                ]
            ),
            "kw": args_to_list(
                [
                    param
                    for param in ctr_signature.parameters.values()
                    if is_optional(param) and not is_bad(param)
                ]
            ),
        }
    else:
        ctr = {}
    if fwd_signature:
        fwd = {
            "pos": args_to_list(
                [
                    param
                    for param in fwd_signature.parameters.values()
                    if is_positional(param) and not is_bad(param)
                ]
            ),
            "kw": args_to_list(
                [
                    param
                    for param in fwd_signature.parameters.values()
                    if is_optional(param) and not is_bad(param)
                ]
            ),
        }
    else:
        fwd = {}

    return {"prefix": prefix, "path": path, "ctr": ctr, "fwd": fwd}


def create_jinja_env():
    env = Environment(
        loader=PackageLoader("model_builder"), autoescape=select_autoescape()
    )

    def type_name(value):
        value = str(value)
        assert value.startswith("<class '"), f"expected {value} to start with <class '"
        return value[8:-2]

    env.globals.update(type_name=type_name)
    return env


def generatev2(path: str) -> str:
    template_args = get_component_template_args_v2(path)
    env = create_jinja_env()
    template = env.get_template("componentv2.py.jinja")
    return template.render(**template_args)


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


class EmptySphinxException(Exception):
    pass


@functools.cache
def sphinxfy(classpath: str) -> str:
    """
    Takes a classpath and process it's docstring into HTML
    strings

    Args:
        classpath (str):

    Raises:
        EmptySphinxException: when the sphinxfy task does not produce any output
    """
    SPHINX_CONF = r"""
extensions = ['sphinx.ext.napoleon', 'sphinx.ext.autodoc', 'sphinx.ext.mathjax']

templates_path = ['templates']
html_static_path = ['static']

html_use_modindex = False
html_use_index = False
html_split_index = False
html_copy_source = False

todo_include_todos = True"""
    HTML_TEMPLATE = r"""<div class="docstring">
    {% block body %} {% endblock %}
</div>"""

    srcdir, outdir = tempfile.mkdtemp(), tempfile.mkdtemp()
    src_base_name, out_base_name = os.path.join(srcdir, "docstring"), os.path.join(
        outdir, "docstring"
    )
    rst_name, out_name = src_base_name + ".rst", out_base_name + ".html"
    docstring = get_class_from_path_string(classpath).__doc__
    with open(rst_name, "w") as filed:
        filed.write(docstring)

    # Setup sphinx configuratio
    confdir = os.path.join(srcdir, "en", "introspect")
    os.makedirs(confdir)
    with open(os.path.join(confdir, "conf.py"), "w") as filed:
        filed.write(SPHINX_CONF)

    # Setup sphixn templates dir
    templatesdir = os.path.join(confdir, "templates")
    os.makedirs(templatesdir)
    with open(os.path.join(templatesdir, "layout.html"), "w") as filed:
        filed.write(HTML_TEMPLATE)
    doctreedir = os.path.join(srcdir, "doctrees")
    confoverrides = {"html_context": {}, "master_doc": "docstring"}
    old_sys_path = list(sys.path)  # Sphinx modifies sys.path
    # Secret Sphinx reference
    # https://www.sphinx-doc.org/en/master/_modules/sphinx/application.html#Sphinx
    sphinx_app = Sphinx(
        srcdir=srcdir,
        confdir=confdir,
        outdir=outdir,
        doctreedir=doctreedir,
        buildername="html",
        freshenv=True,
        confoverrides=confoverrides,
        status=None,  # defines where to log, default was os.stdout
        warning=None,  # defines where to log errors, default was os.stderr
        warningiserror=False,
    )
    sphinx_app.build(False, [rst_name])
    sys.path = old_sys_path

    if os.path.exists(out_name):
        with open(out_name, "r") as f:
            output = f.read()
            # Remove spurious \(, \), \[, \].
            output = (
                output.replace(r"\(", "")
                .replace(r"\)", "")
                .replace(r"\[", "")
                .replace(r"\]", "")
            )
            output = re.sub("<img[^>]*>", "", output)
        return output
    else:
        raise EmptySphinxException()


if __name__ == "__main__":
    import sys

    template = sys.argv[1]
    if template == "component":
        compnames = sys.argv[2:]
        for compname in compnames:
            print(generatev2(compname))
    elif template == "base":
        bundle = generate_bundlev2()
        print(bundle)
        sys.exit(0)
