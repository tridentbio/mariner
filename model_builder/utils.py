from collections.abc import Mapping
from typing import Any, Callable, Dict, Sequence, Union

import numpy as np
import torch
from humps import camel
from pydantic import BaseModel
from torch_sparse import SparseTensor

from model_builder.storage import BaseStorage


class DataInstance(BaseStorage):
    """DataInstance basically works like a map/storage. It works
    through a structure similar to a python dict with some more
    features to support pytorch operations. This way it is possible
    to support types such as tensors, `pytorch_geometric.Data` and
    other data types used in models.

    For information about the methods see:
    https://docs.python.org/3/reference/datamodel.html

    Args:
        y (Any): The target value for that instance.
        **kwargs: Any arg passed via kwargs will become an
            attribute of the instance.

    Example:
    >>> data = DataInstance()
    >>> data.x = torch.tensor([1.76])
    """

    def __init__(self, y=None, **kwargs):

        self.__dict__["_store"] = BaseStorage(_parent=self)

        if y is not None:
            self.y = y

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return self._store[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._store[key] = value

    def __delitem__(self, key: str) -> None:
        if key in self._store:
            del self._store[key]

    def __getattr__(self, key: str) -> Any:
        if "_store" not in self.__dict__:
            raise RuntimeError
        return getattr(self._store, key)

    def __setattr__(self, key: str, value: Any) -> None:
        setattr(self._store, key, value)

    def __delattr__(self, key: str, value: Any) -> None:
        delattr(self._store, key)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = [size_repr(k, v, indent=2) for k, v in self._store.items()]
        info = ",\n".join(info)
        return f"{cls}(\n{info}\n)"


def recursive_apply_(data: Any, function: Callable) -> None:
    """Recursively apply a function to the data structures that
    have been passed in.

    Args:
        data (Any): Data structure to apply the function.
        function (Callable): Function that will be applied to the data.
    """
    if isinstance(data, torch.Tensor):
        function(data)
        return

    if isinstance(data, tuple) and hasattr(data, "_fields"):
        for value in data:
            recursive_apply_(value, function)
        return

    if isinstance(data, Sequence) and not isinstance(data, str):
        for value in data:
            recursive_apply_(value, function)
        return

    if isinstance(data, Mapping):
        for value in data.values:
            recursive_apply_(value, function)
        return

    function(data)


def size_repr(key: Any, value: Any, indent: int = 0) -> str:
    """Creates a string representation varying according to
    the passed data structure.

    Args:
        key (Any): Store key to be used to create the repr.
        value (Any): Store value used to represent the data.
        indent (int, optional): Level of indentations used if
            necessary. Defaults to 0.

    Returns:
        str: Representation of data in string form.
    """
    pad = " " * indent

    if isinstance(value, torch.Tensor) and value.dim() == 0:
        out = value.item()

    elif isinstance(value, torch.Tensor):
        out = str(list(value.size()))

    elif isinstance(value, np.ndarray):
        out = str(list(value.shape))

    elif isinstance(value, SparseTensor):
        out = str(value.sizes())[:-1] + f", nnz={value.nnz()}]"

    elif isinstance(value, str):
        out = f"'{value}'"

    elif isinstance(value, Sequence):
        out = str(len(value))

    elif isinstance(value, Mapping) and len(value) == 0:
        out = "{}"

    elif (
        isinstance(value, Mapping)
        and len(value) == 1
        and not isinstance(list(value.values())[0], Mapping)
    ):
        lines = [size_repr(k, v, 0) for k, v in value.items()]
        out = "{ " + ", ".join(lines) + " }"

    elif isinstance(value, Mapping):
        lines = [size_repr(k, v, indent + 2) for k, v in value.items()]
        out = "{\n" + ",\n".join(lines) + "\n" + pad + "}"

    else:
        out = str(value)

    key = str(key).replace("'", "")
    return f"{pad}{key}={out}"


def split_module_export(classpath: str) -> tuple[str, str]:
    words = classpath.split(".")
    module = ".".join(words[:-1])
    export = words[-1]
    return module, export


def get_class_from_path_string(pathstring: str) -> Any:
    """Dynamically import a class from a string path.

    Args:
        pathstring (str): String path to the class.

    Returns:
        Any: The class.
    """
    module_name, export = split_module_export(pathstring)
    code = f"""
from {module_name} import {export}
references['cls'] = {export}
"""
    references: Dict[str, Any] = {}
    exec(code, globals(), {"references": references})
    return references["cls"]


def unwrap_dollar(value: str) -> tuple[str, bool]:
    """
    Takes a string and remove it's reference indicators: $ or ${...}
    Returns the string unwrapped (if it was a reference) and `is_reference` boolean
    """
    if value.startswith("${") and value.endswith("}"):
        return value[2:-1], True
    elif value.startswith("$"):
        return value[1:], True
    return value, False


def get_references_dict(
    forward_args_dict: dict[str, Any]
) -> dict[str, Union[str, list[str]]]:
    result = {}
    for key, value in forward_args_dict.items():
        if isinstance(value, list):
            result[key] = []
            for item in value:
                ref, is_ref = unwrap_dollar(item)
                if is_ref:
                    result[key].append(ref)
        elif isinstance(value, str):
            ref, is_ref = unwrap_dollar(value)
            if is_ref:
                result[key] = ref
        else:
            continue
    return result


def get_ref_from_input(value: str, input: DataInstance):
    attribute, attribute_accessors = (
        value.split(".")[0],
        value.split(".")[1:],
    )
    if attribute == "input":
        return input["".join(attribute_accessors)]  # $inputs.x is accessed as data['x']
    value = input[attribute]
    if len(attribute_accessors) == 0:
        return value
    # TODO: fix edges_weight and edge_attr usage
    if "edge_weight" in attribute_accessors or "edge_attr" in attribute_accessors:
        return None
    value = value[0]  # hack to flatten array of DataBatches (output of Featurizer)
    for attr in attribute_accessors:
        value = getattr(value, attr)
    return value


def collect_args(
    input: DataInstance, args_dict: dict[str, Any]
) -> Union[list, dict, Any]:
    """
    Takes a DataInstance object and a dictionary of args to retrive
    and returns a dictinoraty of resolved arguments.
    Each arg value may be a reference. in such case, the arg value must be
    retrieved from `input`. Otherwise, it's raw value that can be returned as is
    """
    result = {}
    for key, value in args_dict.items():
        if value is None:
            continue
        if isinstance(value, list):
            value_ = []
            for item in value:
                item_value, item_is_ref = unwrap_dollar(item)
                if item_is_ref:
                    value_.append(get_ref_from_input(item_value, input))
                else:
                    value_.append(item_value)
            result[key] = value_
        elif isinstance(value, str):
            if not value:
                continue
            value, is_ref = unwrap_dollar(value)
            if is_ref:
                result[key] = get_ref_from_input(value, input)
            else:
                result[key] = value
        else:
            raise ValueError("args_dict must be list or str")
    return result


class CamelCaseModel(BaseModel):
    class Config:
        alias_generator = camel.case
        allow_population_by_field_name = True
        allow_population_by_alias = True
        underscore_attrs_are_private = True
