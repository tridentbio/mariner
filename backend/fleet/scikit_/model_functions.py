from mlflow.entities.model_registry.model_version import ModelVersion
from pandas import DataFrame

from fleet.base_schemas import BaseModelFunctions
from fleet.scikit_.schemas import SciKitModelSpec, SciKitTrainingConfig


class SciKitFunctions(BaseModelFunctions):
    def __init__(self):
        ...

    def train(
        self, spec: SciKitModelSpec, params: SciKitTrainingConfig, dataset: DataFrame
    ) -> None:
        ...

    def test(self) -> None:
        ...

    def load(self) -> None:
        ...

    def log_models(
        self, mlflow_experiment_id: str, mlflow_model_name: str
    ) -> ModelVersion:
        ...
