"""
Defines the integer featurizer
"""
from typing import Union

import torch
from typing_extensions import override

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

    def __init__(self):
        super().__init__()

    def __call__(self, input_: str) -> torch.Tensor:
        return self.featurize(input_)

    def featurize(self, input_: Union[str, int, float]):
        if isinstance(input_, float):
            input_ = int(input_)
        if str(input_) not in self.classes:
            raise RuntimeError(
                f"Element {input_} of type {input_.__class__} is not defined"
                f" in the classes dictionary {self.classes}"
            )
        return torch.Tensor([self.classes[str(input_)]]).long()

    def unfeaturize(self, input_: torch.Tensor):
        idx = int(input_.item())
        if idx not in self.reversed_classes:
            raise RuntimeError(
                f"Element {input_} is not defined in the"
                "classes dictionary {self.classes}"
            )
        return self.reversed_classes[idx]

    @override
    def set_from_model_schema(self, config, dataset_config, deps):
        input_ = deps[0]  # featurizer has a single argument to __call__
        # Get column information from schema
        column_info = get_column_config(config, input_)
        # Handle missing column information
        if not column_info:
            raise RuntimeError(f"Column {input_} was not found in the config columns")

        from fleet.dataset_schemas import CategoricalDataType

        # Handle column info not being from categorical

        if not isinstance(column_info.data_type, CategoricalDataType):
            raise DataTypeMismatchException(
                "expecteing CategoricalDataType, but found"
                f"{column_info.data_type.__class__}",
                expected=CategoricalDataType,
                got_item=column_info.data_type,
            )
        self.classes = column_info.data_type.classes
        self.reversed_classes = {value: key for key, value in self.classes.items()}
