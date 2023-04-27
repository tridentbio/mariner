"""
This package contain utility classes and functions used in the model builder
"""
from collections.abc import Mapping
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch
from humps import camel
from pydantic import BaseModel
from torch_sparse import SparseTensor

from fleet.model_builder.storage import BaseStorage


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
        **kwargs: Any argument passed via kwargs will become an
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

    def __delattr__(self, key: str) -> None:
        delattr(self._store, key)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info_parts = [size_repr(k, v, indent=2) for k, v in self._store.items()]
        info = ",\n".join(info_parts)
        return f"{cls}(\n{info}\n)"


def size_repr(key: Any, value: Any, indent: int = 0) -> str:
    """Creates a string representation varying according to
    the passed data structure.

    Args:
        key (Any): Store key to display in ``repr``.
        value (Any): Store value used to represent the data.
        indent (int, optional): Level of indentations used if
            necessary. Defaults to 0.

    Returns:
        str: Representation of data in string form.
    """
    pad = " " * indent

    if isinstance(value, torch.Tensor) and value.dim() == 0:
        out = str(value.item())

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
    """Splits a complete class path such as "A.B.foo" into it's
    module ("A.B") and definition ("foo") strings

    Args:
        classpath: the complete class path, .e.g. ``"torch.nn.Linear"``

    Returns:
        (module, definition) tuple of strings
    """
    words = classpath.split(".")
    module = ".".join(words[:-1])
    export = words[-1]
    return module, export


def get_class_from_path_string(pathstring: str) -> type:
    """Dynamically import a class from a string path.

    Args:
        pathstring (str): String path to the class.

    Returns:
        Any: The class.
    """
    return __import__(pathstring)


def unwrap_dollar(value: str) -> tuple[str, bool]:
    """
    Takes a string and remove its reference indicators: $ or ${...}
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
    """Gets a dictionary with the same shape of forward_args_dict
    but with all string values representing reference mapped to the
    reference name.

    Args:
        forward_args_dict: dict with forward arguments input.

    Returns:
        dictionary mapping forward argument keys to schema references.
    """

    # Init a dict
    result: dict[str, Union[str, list[str]]] = {}

    # Loop through the forward args dict
    for key, value in forward_args_dict.items():
        # Handle list instances
        if isinstance(value, list):
            result_key = []
            for item in value:
                ref, is_ref = unwrap_dollar(item)
                if is_ref:
                    result_key.append(ref)
            result[key] = result_key
        elif isinstance(value, str):
            ref, is_ref = unwrap_dollar(value)
            if is_ref:
                result[key] = ref

    return result


def get_ref_from_data_instance(accessor_str: str, input: DataInstance):
    """Gets the value of a DataInstance

    Used by model builder to interpret the forward_args.
    When ``accessor_str`` contains '.', it's value  is splitted on
    the first occurrence, e.g:
        - ``"foo.a.b.c"`` is splitted to ``["foo", "a.b.c"]``
    The first string of the tuple is used as a key of the ``input`` and
    the last one is used to access a possibly nested attribute in the value
    found in ``input``.

    Args:
        accessor_str: string representing attribute to get, e.g. "mol_featurizer.x", "mwt"
        input: data instance with collected layer/featurizer outputs, inputs
    and outputs
    """
    attribute, attribute_accessors = (
        accessor_str.split(".")[0],
        accessor_str.split(".")[1:],
    )
    if attribute == "input":
        return input["".join(attribute_accessors)]  # $inputs.x is accessed as data['x']
    value = input[attribute]
    if len(attribute_accessors) == 0:
        return value

    for attr in attribute_accessors:
        value = getattr(value, attr)
    return value


def get_ref_from_input(
    accessor_str: str, input: Union[DataInstance, List[DataInstance]]
):
    """Gets the accessor_str of a DataInstance or a batch of DataInstance's


    Args:
        accessor_str: string represented accessor_str to get, e.g. "mol_featurizer.x", "mwt".
        input(Union[DataInstance, List[DataInstance]):
    """
    if isinstance(input, list):
        values = [get_ref_from_data_instance(accessor_str, item) for item in input]
        assert len(values) > 0, "failed to get values from input"
        if isinstance(values[0], torch.Tensor):
            return torch.cat(values)  # type: ignore

    else:
        return get_ref_from_data_instance(accessor_str, input)


def collect_args(
    input_: DataInstance, args_dict: Dict[str, Any]
) -> Union[list, dict, Any]:
    """Takes a batch of DataInstance objects and a dictionary of arguments to retrieve
    and returns a dictionary of resolved arguments.

    Each argument accessor_str may be a reference. In such case, the argument accessor_str must be
    retrieved from `input`. Otherwise, it's raw accessor_str that can be returned as is.
    """
    result: Dict[str, Union[Any, str, None]] = {}
    for key, value in args_dict.items():
        if value is None:
            continue
        if isinstance(value, list):
            value_ = []
            for item in value:
                item_value, item_is_ref = unwrap_dollar(item)
                if item_is_ref:
                    value_.append(get_ref_from_input(item_value, input_))
                else:
                    value_.append(item_value)
            result[key] = value_
        elif isinstance(value, str):
            if not value:
                continue
            value, is_ref = unwrap_dollar(value)
            if is_ref:
                result[key] = get_ref_from_input(value, input_)
            else:
                result[key] = value
        else:
            raise ValueError("args_dict must be list or str")
    return result


class CamelCaseModel(BaseModel):
    """Specialize pydantic models to work with camel case when working with json."""

    class Config:
        """Configures the wrapper class to work as intended."""

        alias_generator = camel.case
        allow_population_by_field_name = True
        allow_population_by_alias = True
        underscore_attrs_are_private = True
