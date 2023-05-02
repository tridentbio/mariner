"""
OneHot custom layer
"""
from functools import reduce
from typing import List, Literal, Union

import torch
from torch import nn
from torch.nn import functional as F

from model_builder.component_builder import AutoBuilder
from model_builder.exceptions import DataTypeMismatchException
from model_builder.model_schema_query import get_column_config
from model_builder.schemas import CategoricalDataType


class OneHot(nn.Module, AutoBuilder):
    """
    A helper layer that outputs the one-hot encoding representation of
    it's categorical inputs
    """

    # this property is only filled on training, when we have the dataset "at hands"
    classes: Union[dict[Union[str, int], int], None] = None

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x1: Union[List[str], List[int]]) -> torch.Tensor:
        """One hot representation of the input

        Args:
            x1: list of tensors

        Returns:
           one hot representation
        """
        assert self.classes, "OneHot layer is missing the classes property set"
        longs = self.serialize(self.classes, x1, "tensor")
        return F.one_hot(longs, num_classes=len(self.classes)).float()

    def set_from_model_schema(self, config, deps):
        """Sets classes dict from the model schema

        Args:
            config (ModelSchema): model schema that provides classes
            deps (list[str]): Name of the column received

        Raises:
            RuntimeError: If some element in deps is not
            found in the model schema
            DataTypeMismatchException: When the element in
            deps is not categorical
        """
        input_ = deps[0]  # this layer has a single arg to forward method
        column_config = get_column_config(config, input_)
        if not column_config:
            raise RuntimeError(f"Column config not found for input {input_}")
        if not isinstance(column_config.data_type, CategoricalDataType):
            raise DataTypeMismatchException(
                f"Expected data type categorical but got {column_config.__class__}",
                expected=CategoricalDataType,
                got_item=column_config,
            )
        self.classes = column_config.data_type.classes

    def serialize(
        self,
        classes: dict,
        xs: List[Union[list, str]],
        returns_type: Literal["tensor", "list"] = "tensor",
    ):
        """Serialize a categorical data into a list of numbers

        e.g.:
            classes: {'a': 0, 'b': 1}
            xs: ['a', 'b', ['a', 'b']]
            -> [0, 1, [0, 1]]

        Args:
            classes (dict): a dict of classes map to numbers
            xs (list): a list of keys of classes
            returns_type (str): return type, either 'tensor' or 'list'

        Returns:
            a list of numbers or a tensor of numbers
        """
        data = reduce(
            lambda acc, cur: (
                acc + self.serialize(classes, cur, "list")
                if isinstance(cur, list)
                else acc + [classes[cur]]
            ),
            xs,
            [],
        )

        if returns_type == "list":
            return data
        elif returns_type == "tensor":
            return torch.Tensor(data).long()
