from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch

T = TypeVar("T")


class BaseFeaturizer(ABC, Generic[T]):
    """Base class for any featurizer."""

    @abstractmethod
    def __call__(self, input: T) -> torch.Tensor:
        ...


class ReversibleFeaturizer(BaseFeaturizer[T], ABC):
    """Base class for reversible featurizers."""

    @abstractmethod
    def undo(self, input: torch.Tensor) -> T:
        ...
