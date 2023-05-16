"""
Abstract classes that provide interfaces for model and training schemas,
as well as the interface for using models.

Classes from this packages should not be instantiated, or used as types.
Instead, the schemas should be implemented in the framework schemas packages,
e.g. torch_.schemas, sklearn_.schemas. Than, in fleet.schemas we have the
FleetModelSpec, which unifies the descriptions of all framework schemas,
and FleetTrainingConfig, that does the same for framework training parameters.
"""
from abc import ABC
from typing import Literal, Union

from mlflow.entities.model_registry.model_version import ModelVersion
from pandas import DataFrame
from pydantic import BaseModel
from typing_extensions import Protocol

from fleet.dataset_schemas import DatasetConfig, TorchDatasetConfig
from fleet.model_builder.schemas import TorchModelSchema
from fleet.model_builder.utils import CamelCaseModel
from fleet.yaml_model import YAML_Model


class BaseFleetModelSpec(CamelCaseModel, ABC, YAML_Model):
    """
    Base class for framework model specs.

    Attributes:
        name: An identifier of the spec.
        framework: The framework this model uses. It should be narrowed to a
            literal type by the subclasses, e.g. 'torch', 'sklearn'.
        dataset: A description of the dataset columns data types and
            transformations. It's possible to extend the dataset config to support
            framework specific things, such it's done with :py:class:`TorchDatasetConfig`, but
            it's not encouraged.
        spec: A description of the model parameters. It should be narrowed to a
            specific pydantic model.
    """

    name: str
    framework: str
    dataset: "DatasetConfig"
    spec: BaseModel


class TorchModelSpec(CamelCaseModel, YAML_Model):
    """
    Concrete implementation of torch model specs.
    TODO: move to fleet.torch_.schemas
    """

    name: str
    framework: Literal["torch"] = "torch"
    spec: TorchModelSchema
    dataset: "TorchDatasetConfig"


class ScikitModelSpec(CamelCaseModel, YAML_Model):
    """
    Concrete implementation of sklearn model specs.
    TODO: move to fleet.sklearn_.schemas
    """

    name: str
    framework: Literal["sklearn"] = "sklearn"
    spec: "TorchModelSchema"
    dataset: "TorchDatasetConfig"


# TOD: move to fleet.schemas
FleetModelSpec = Union[TorchModelSpec, ScikitModelSpec]


class BaseModelFunctions(Protocol):
    """
    Interface for training, testing and using a model for a specific framework.
    """

    def train(
        self,
        *,
        dataset: DataFrame,
        spec: BaseFleetModelSpec,
        params: BaseModel,
        datamodule_args: Union[None, dict] = None,
    ) -> None:
        """Trains a model.

        Trains the model described by `spec` with the `dataset` and training
        parameters in `params`. To effectively use this model, once `train(...)`
        finishes, the user should call `log_models(...)`

        Args:
            dataset: :py:class:`DataFrame` with the data as described in `spec.dataset`
            spec: Specification of the model parameters.
            params: Training/Fitting parameters.
        """

    def test(self) -> None:
        """
        Tests a model.

        TODO: Add arguments and implement method in subclasses.
        """

    def log_models(
        self, mlflow_experiment_id: str, mlflow_model_name: str
    ) -> ModelVersion:
        """Publishes the model to mlflow.

        Should only be called after :meth:`train`.

        Args:
            mlflow_experiment_id: the id of the mlflow experiment to publish the trained model.
            mlflow_model_name: the name of the trained models

        Returns:
            Returns the :py:class:`ModelVersion`
        """
        ...

    def load(self) -> None:
        """Loads a model to be tested."""
