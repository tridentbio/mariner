from typing import Union

import torch

from model_builder.component_builder import AutoBuilder
from model_builder.exceptions import DataTypeMismatchException
from model_builder.featurizers.base_featurizers import ReversibleFeaturizer
from model_builder.model_schema_query import get_column_config
from model_builder.schemas import CategoricalDataType


class IntegerFeaturizer(ReversibleFeaturizer[Union[str, int]], AutoBuilder):
    classes: dict[Union[str, int], int]

    def __call__(self, input) -> torch.Tensor:
        if input not in self.classes:
            raise RuntimeError(
                f"Element {input} is not defined in the classes dictionary {self.classes}"
            )
        return torch.Tensor([self.classes[input]]).long()

    def undo(self, input: torch.Tensor):
        idx = int(input.item())
        if idx not in self.reversed_classes:
            raise RuntimeError(
                f"Element {input} is not defined in the classes dictionary {self.classes}"
            )
        return self.reversed_classes[idx]

    def set_from_model_schema(self, config, inputs):
        input = inputs[0]  # featurizer has a single argument to __call__
        # Get column information from schema
        column_info = get_column_config(config, input)
        # Handle missing column information
        if not column_info:
            raise RuntimeError(f"Column {input} was not found in the config columns")
        # Handle column info not being from categorical
        if not isinstance(column_info, CategoricalDataType):
            raise DataTypeMismatchException(
                f"expecteing CategoricalDataType, but found {column_info.__class__}",
                expected=CategoricalDataType,
                got_item=column_info,
            )
        self.classes = column_info.classes
        self.reversed_classes = {value: key for key, value in self.classes.items()}
