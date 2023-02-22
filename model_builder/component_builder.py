"""
Base classes to help building components using context

E.g. base class for components that need schema info to
auto fill some argument
"""
from abc import ABC, abstractmethod

from model_builder.schemas import ModelSchema


class AutoBuilder(ABC):
    """Base class for components that need the model schema to fill it's parameters.
    Classes that inherit the AutoBuilder must implement a method that sets attributes
    gotten from the model schema
    """

    @abstractmethod
    def set_from_model_schema(self, config: ModelSchema, deps: list[str]):
        """Method to implement argument filling from ModelSchema

        Must be overriden

        Args:
            config: ModelSchema instance
            deps: node names that are previous to this one
        """
        ...
