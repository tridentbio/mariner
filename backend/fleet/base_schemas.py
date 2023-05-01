from abc import ABC
from typing import Any

from mlflow.entities.model_registry.model_version import ModelVersion
from pandas import DataFrame
from pydantic import BaseModel
from typing_extensions import Protocol

from fleet.dataset_schemas import DatasetConfig
from fleet.model_builder.utils import CamelCaseModel
from fleet.yaml_model import YAML_Model


class BaseFleetModelSpec(CamelCaseModel, ABC, YAML_Model):
    name: str
    framework: str
    dataset: DatasetConfig
    spec: Any


class BaseModelFunctions(Protocol):
    def train(
        self, *, dataset: DataFrame, spec: BaseFleetModelSpec, params: BaseModel
    ) -> None:
        ...

    def test(self) -> None:
        ...

    def log_models(
        self, mlflow_experiment_id: str, mlflow_model_name: str
    ) -> ModelVersion:
        ...

    def load(self) -> None:
        ...
