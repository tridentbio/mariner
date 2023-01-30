"""
OneHot custom layer
"""
from typing import Union

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
        super(nn.Module).__init__()

    def forward(self, x1: Union[list[str], list[int]]) -> torch.Tensor:
        """One hot representation of the input

        Args:
            x1: list of tensors

        Returns:
           one hot representation
        """
        assert self.classes, "OneHot layer is missing the classes property set"
        longs = torch.Tensor([self.classes[x] for x in x1]).long()
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
