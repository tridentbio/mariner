from collections.abc import Mapping
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch_sparse import SparseTensor


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

    try:
        function(data)
    except:
        pass


def recursive_apply(data: Any, function: Callable) -> Any:
    """Recursively applies a function to the passed data structure and
    returns the equivalent type resulting from applying the function.

    Args:
        data (Any): Data structure to apply the function.
        function (Callable): Function that will be applied to the data.

    Returns:
        Any: Data resulting from the application function.
    """
    if isinstance(data, torch.Tensor):
        return function(data)

    if isinstance(data, torch.nn.utils.rnn.PackedSequence):
        return function(data)

    if isinstance(data, tuple) and hasattr(data, "_fields"):
        return type(data)(*(recursive_apply(d, function) for d in data))

    if isinstance(data, Sequence) and not isinstance(data, str):
        return [recursive_apply(d, function) for d in data]

    if isinstance(data, Mapping):
        return {key: recursive_apply(data[key], function) for key in data}

    try:
        return function(data)
    except:
        return data


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
