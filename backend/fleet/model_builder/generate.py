"""
Layers and Featurizers code generation related code
"""
import functools
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from inspect import Parameter, Signature, signature
from typing import Any, List, Literal, Optional, Union

from humps import camel
from jinja2 import Environment, PackageLoader, select_autoescape
from sphinx.application import Sphinx

from fleet.model_builder.utils import get_class_from_path_string


@dataclass
class Layer:
    """
    Represents resource to be inspected to generate code
    """

    name: str
    type: Literal["layer", "featurizer"]


featurizers = [
    Layer(name=name, type="featurizer")
    for name in [
        "fleet.model_builder.featurizers.MoleculeFeaturizer",
        "fleet.model_builder.featurizers.IntegerFeaturizer",
        "fleet.model_builder.featurizers.DNASequenceFeaturizer",
        "fleet.model_builder.featurizers.RNASequenceFeaturizer",
        "fleet.model_builder.featurizers.ProteinSequenceFeaturizer",
    ]
]

layers = [
    Layer(name=name, type="layer")
    for name in [
        "fleet.model_builder.layers.OneHot",
        "fleet.model_builder.layers.GlobalPooling",
        "fleet.model_builder.layers.Concat",
        "fleet.model_builder.layers.AddPooling",
        "torch.nn.Linear",
        "torch.nn.Sigmoid",
        "torch.nn.ReLU",
        "torch_geometric.nn.GCNConv",
        # "model_builder.featurizers.FromSmiles",
        "torch.nn.Embedding",
        "torch.nn.TransformerEncoderLayer",
    ]
]


python_primitives_reprs = {
    int: "int",
    float: "float",
    bool: "bool",
    str: "string",
}


def get_component_constructor_signature(
    class_def: Any,
) -> Optional[Signature]:
    """
    Gets the __init__ method signature

    Args:
        class_def (Any): Class to inspect

    Returns:
        Optional[Signature]:
            The signature of the method or None if the method does not exist
    """
    return get_component_signature(class_def, "__init__")


def get_component_forward_signature(class_def: Any) -> Optional[Signature]:
    """
    Gets the forward ( or __call__ for featurizers ) Signature

    Args:
        class_def (Any): Class to inspect

    Returns:
        Optional[Signature]:
            The signature of the method or None if the method does not exist
    """
    if "forward" in dir(class_def):
        return get_component_signature(class_def, "forward")
    elif "__call__" in dir(class_def):
        return get_component_signature(class_def, "__call__")
    return None


def get_component_signature(
    class_def: Any, method_name: str
) -> Optional[Signature]:
    """Gets the signature of a method of a class

    Args:
        class_def (Any): Class to inspect
        method_name (str): Class method to inspect name

    Raises:
        ValueError: If the method has no __code__ attribute

    Returns:
        Optional[Signature]:
            The signature of the method or None if the method does not exist
    """
    if method_name not in dir(class_def):
        raise ValueError(f"{method_name} not found in class_def")
    method = getattr(class_def, method_name)
    return signature(method)


def is_bad(parameter: Parameter) -> bool:
    """Returns True if the parameter is bad

    Bad parameters are those that are not positional or keyword only

    Args:
        parameter (Parameter): the parameter

    Returns:
        bool: True if the parameter is bad else False
    """
    return parameter.annotation == Parameter.empty


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


def is_primitive(val: Any):
    """Checks if val is instance of valid primitive

    Args:
        val: value to be checked
    """
    return isinstance(val, (int, float, str, bool))


def args_to_list(params: List[Parameter]) -> List[tuple[str, Any, Any]]:
    """Returns a list of tuples with the name, type and default value of the parameters

    Args:
        params (List[Parameter]): list of parameters

    Returns:
        List[tuple[str, Any, Any]]:
            list of tuples with the name, type and default value of the parameters
    """
    return [
        (
            param.name,
            str(param.annotation),
            param.default if is_primitive(param.default) else None,
        )
        for param in params
    ]


def get_component_template_args_v2(layer: Layer):
    """
    Gets the dictionary with the arguments passed to jinja template engine to
    render the autogenerated code for a component (layer or featurizer)

    Below is a description of the dictionary built:
    - "prefix"
    - "path"
    - "ctr": A description of the constructor signature of the class referenced by path
      - "pos": (arg_name, arg_type, None) The positional (required) arguments
      - "kw": (arg_name, arg_type, arg_default) The keyword (optional) arguments.
    - "fwd": Describes call signature. If no forward found, use __call__ instead
      - "pos":
      - "kw":
    """
    path = layer.name
    path_parts = path.split(".")
    component_name: Union[None, str] = None
    if "_" in path_parts[-1]:
        component_name = camel.case(path_parts[-1])
    else:
        component_name = path_parts[-1]
    prefix = camel.case(path_parts[0]) + component_name
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
    return {
        "prefix": prefix,
        "path": path,
        "ctr": ctr,
        "fwd": fwd,
        "type": layer.type,
    }


def create_jinja_env() -> Environment:
    """Creates a jinja environment with the custom filters and globals

    Returns:
        Environment: jinja environment
    """
    env = Environment(
        loader=PackageLoader("fleet.model_builder"),
        autoescape=select_autoescape(),
    )

    def type_name(value):
        value = str(value)
        if value.startswith("<class '"):
            return f"{value[8:-2]}"
        elif value.startswith("typing."):
            return value.replace("typing.", "")

    env.globals.update(type_name=type_name)
    return env


def generatev2(path: str) -> str:
    """Generates the bundle python code using componentv2.py.jinja as a template

    Returns:
        str: python code for the bundle.py file
    """
    template_args = get_component_template_args_v2(path)
    env = create_jinja_env()
    template = env.get_template("componentv2.py.jinja")
    return template.render(**template_args)


def generate_bundlev2() -> str:
    """Generates the bundle python code using base.py.jinja as a template

    Returns:
        str: python code for the bundle.py file
    """
    env = create_jinja_env()
    schemas_template = env.get_template("base.py.jinja")
    layer_template_args = [
        get_component_template_args_v2(layer) for layer in layers
    ]
    featurizer_template_args = [
        get_component_template_args_v2(featurizer)
        for featurizer in featurizers
    ]
    return schemas_template.render(
        layer_components=layer_template_args,
        featurizer_components=featurizer_template_args,
    )


class EmptySphinxException(Exception):
    """Raised when sphix outputs empty docs"""


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
    sphinx_conf = r"""
extensions = ['sphinx.ext.napoleon', 'sphinx.ext.autodoc', 'sphinx.ext.mathjax']

templates_path = ['templates']
html_static_path = ['static']

html_use_modindex = False
html_use_index = False
html_split_index = False
html_copy_source = False

todo_include_todos = True"""
    html_template = r"""<div class="docstring">
    {% block body %} {% endblock %}
</div>"""

    srcdir, outdir = tempfile.mkdtemp(), tempfile.mkdtemp()
    src_base_name, out_base_name = os.path.join(
        srcdir, "docstring"
    ), os.path.join(outdir, "docstring")
    rst_name, out_name = src_base_name + ".rst", out_base_name + ".html"
    docstring = get_class_from_path_string(classpath).__doc__
    if not docstring:
        return ""

    with open(rst_name, "w", encoding="utf-8") as file:
        file.write(docstring)

    # Setup sphinx configuratio
    confdir = os.path.join(srcdir, "en", "introspect")
    os.makedirs(confdir)
    with open(os.path.join(confdir, "conf.py"), "w", encoding="utf-8") as file:
        file.write(sphinx_conf)

    # Setup sphixn templates dir
    templatesdir = os.path.join(confdir, "templates")
    os.makedirs(templatesdir)
    with open(
        os.path.join(templatesdir, "layout.html"), "w", encoding="utf-8"
    ) as file:

        file.write(html_template)
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
        with open(out_name, "r", encoding="utf-8") as file:
            output = file.read()
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
    TEMPLATE = sys.argv[1]
    if TEMPLATE == "component":
        compnames = sys.argv[2:]
        for compname in compnames:
            out = generatev2(compname)
            print(out)
    elif TEMPLATE == "base":
        bundle = generate_bundlev2()
        print(bundle)
        sys.exit(0)
