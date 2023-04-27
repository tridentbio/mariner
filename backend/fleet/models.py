"""
The fleet.models module is responsible to export abstractions
around different machine learning algorithms provided by 
underlying packages such as torch and sci-kit
"""

from typing import Annotated, Generic, TypeVar, Union

from pydantic import BaseModel, Field
from typing_extensions import Protocol

from fleet.model_builder.schemas import ModelSchema
from fleet.model_builder.utils import CamelCaseModel

# --------------------------------- #
# Model Specification Definitions   #
# --------------------------------- #


class TorchModelSpec(BaseModel):
    framework = "torch"
    spec: ModelSchema


class ScikitModelSpec(BaseModel):
    framework = "scikit"
    ...


# ---------------------------------- #
# Training Request Configuration     #
# ---------------------------------- #


class MonitoringConfig(CamelCaseModel):
    """
    Configures model checkpointing
    """

    metric_key: str
    mode: str


class EarlyStoppingConfig(CamelCaseModel):
    """
    Configures earlystopping of training
    """

    metric_key: str
    mode: str
    min_delta: float = 5e-2
    patience: int = 5
    check_finite: bool = False


class TorchTrainingSpec(BaseModel):
    epochs: int
    batch_size: Union[None, int] = None
    checkpoint_config: MonitoringConfig


class SciKitTrainingSpec(BaseModel):
    ...


class TorchTrainingRequest(BaseModel):
    framework = "torch"
    spec: TorchTrainingSpec


class ScikitTrainingRequest(BaseModel):
    framework = "scikit"
    spec: SciKitTrainingSpec


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
