"""
The fleet.models module is responsible to export abstractions
around different machine learning algorithms provided by 
underlying packages such as torch and sci-kit
"""

from typing import Annotated, Generic, TypeVar, Union
from typing_extensions import Protocol
from pydantic import BaseModel, Field


# --------------------------------- #
# Model Specification Definitions   #
# --------------------------------- #


class TorchModelSpec(BaseModel):
    framework = "torch"
    ...


class ScikitModelSpec(BaseModel):
    framework = "scikit"
    ...


# The ModelSpec and ModelSpecType is used to generalize the Runner interface
ModelSpec = Annotated[
    Union[TorchModelSpec, ScikitModelSpec], Field(discriminator="framework")
]


# contravariant allows to apply funcs of a supertype to a subtype
# covariant allows to apply funcs of a subtype to a supertype
# We want to apply some f(models: List[ModelSpec]) to List[TorchModelSpec] and List[SciKitModelSpec]
ModelSpecType = TypeVar(
    name="ModelSpecType", bound=ModelSpec, contravariant=True, covariant=False
)

# This is a generic interface for runners, each abstracting a different framework
# training interface


class Runner(Protocol, Generic[ModelSpecType]):
    """
    A generic interface for model spec trainers
    """

    def train(self, model: ModelSpecType):
        """
        Trains a model
        """


# --------------------------------- #
#  Model Training implementations   #
# --------------------------------- #


class TorchRunner(Runner[TorchModelSpec]):
    def train(self, model: TorchModelSpec):
        # TODO
        return super().train(model)


class ScikitRunner(Runner[ScikitModelSpec]):
    def train(self, model: ScikitModelSpec):
        # TODO
        return super().train(model)
