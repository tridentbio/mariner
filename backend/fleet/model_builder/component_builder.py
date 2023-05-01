"""
Base classes to help building components using context

E.g. base class for components that need schema info to
auto fill some argument
"""
import typing
from abc import ABC, abstractmethod

if typing.TYPE_CHECKING:
    from fleet.model_builder.schemas import TorchModelSchema
    from fleet.torch_.schemas import TorchDatasetConfig


class AutoBuilder(ABC):
    """Base class for components that need the model schema to fill it's parameters.
    Classes that inherit the AutoBuilder must implement a method that sets attributes
    gotten from the model schema
    """

    @abstractmethod
    def set_from_model_schema(
        self,
        config: "TorchModelSchema",
        dataset_config: typing.Union[None, "TorchDatasetConfig"] = None,
        deps: list[str] = [],
    ):
        """Method to implement argument filling from ModelSchema

        Must be overriden

        Args:
            config: ModelSchema instance
            deps: node names that are previous to this one
        """
        ...
