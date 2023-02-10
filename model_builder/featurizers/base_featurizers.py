"""Base class for featurizers
"""
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import torch

T = TypeVar("T")


class BaseFeaturizer(ABC, Generic[T]):
    """Base class for any featurizer"""

    @abstractmethod
    def __call__(self, input_: T) -> torch.Tensor:
        """Featurizes a column into a tensor



        Args:
            input_: value to be featurizer

        Returns:
            [TODO:description]
        """


class ReversibleFeaturizer(BaseFeaturizer[T]):
    """Base class for reversible featurizers."""

    @abstractmethod
    def featurize(self, input_: T) -> torch.Tensor:
        """Featurize a sequence

        Args:
            input_: tensor

        Returns:
            the return value has type equals the column featurized
        """

    @abstractmethod
    def unfeaturize(self, input_: torch.Tensor) -> T:
        """Inverse of the featurization

        Args:
            input_: tensor

        Returns:
            the return value has type equals the column featurized
        """
