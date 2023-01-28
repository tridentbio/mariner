from abc import ABC, abstractmethod

from model_builder.schemas import ModelSchema


class AutoBuilder(ABC):
    """Base class for components that need the model schema to fill it's parameters.
    Classes that inherit the AutoBuilder must implement a method that sets attributes
    gotten from the model schema
    """

    @abstractmethod
    def set_from_model_schema(self, config: ModelSchema, deps: list[str]):
        ...
