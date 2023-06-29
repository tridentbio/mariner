"""
Defines the integer featurizer
"""
from collections.abc import Iterable
from typing import Union

import numpy as np
import torch
from typing_extensions import override

from fleet import data_types
from fleet.model_builder.component_builder import AutoBuilder
from fleet.model_builder.exceptions import DataTypeMismatchException
from fleet.model_builder.featurizers.base_featurizers import (
    ReversibleFeaturizer,
)
from fleet.model_builder.model_schema_query import get_column_config


class IntegerFeaturizer(ReversibleFeaturizer[Union[str, int]], AutoBuilder):
    """
    The integer featurizer

    Featurizes categorical data type columns to scalar tensors with dtype long
    """

    classes: dict[Union[str, int], int]
    reversed_classes: dict[int, Union[str, int]]

    def __call__(self, input_: str) -> np.ndarray:
        return self.featurize(input_)

    def featurize(self, input_: Union[str, int, float]):
        if not isinstance(input_, str) and isinstance(input_, Iterable):
            return np.array(
                [self.featurize(i) for i in input_], dtype=np.int64
            )
        elif isinstance(input_, float):
            input_ = int(input_)
        elif str(input_) not in self.classes:
            raise RuntimeError(
                f"Element {input_} of type {input_.__class__} is not defined"
                f" in the classes dictionary {self.classes}"
            )
        else:
            return self.classes[str(input_)]

    def unfeaturize(self, input_: torch.Tensor):
        idx = int(input_.item())
        if idx not in self.reversed_classes:
            raise RuntimeError(
                f"Element {input_} is not defined in the"
                "classes dictionary {self.classes}"
            )
        return self.reversed_classes[idx]

    @override
    def set_from_model_schema(
        self, config=None, dataset_config=None, deps=None
    ):
        if not deps or len(deps) == 0:
            raise ValueError("deps cannot be None")
        if not dataset_config:
            raise ValueError("dataset_config cannot be None")
        input_ = deps[0]  # featurizer has a single argument to __call__
        # Get column information from schema
        column_info = get_column_config(dataset_config, input_)
        # Handle missing column information
        if not column_info:
            raise RuntimeError(
                f"Column {input_} was not found in the config columns"
            )

        # Handle column info not being from categorical
        if not isinstance(
            column_info.data_type, data_types.CategoricalDataType
        ):
            raise DataTypeMismatchException(
                "expecteing CategoricalDataType, but found"
                f"{column_info.data_type.__class__}",
                expected=data_types.CategoricalDataType,
                got_item=column_info.data_type,
            )
        self.classes = column_info.data_type.classes
        self.reversed_classes = {
            value: key for key, value in self.classes.items()
        }
