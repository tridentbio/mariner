"""
Actors for Training, validation and testing models.
"""
from typing import List, Union

import ray
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.loggers.logger import Logger

from fleet.model_functions import fit
from mariner.schemas.dataset_schemas import Dataset
from mariner.schemas.experiment_schemas import Experiment, TorchTrainingRequest
from mariner.schemas.model_schemas import ModelVersion
from mariner.train.custom_logger import MarinerLogger


@ray.remote
class TrainingActor:
    """Runs training with needed mlflow and custom logging"""

    checkpoint_callback: ModelCheckpoint
    early_stopping_callback: Union[EarlyStopping, None] = None
    dataset: Dataset
    modelversion: ModelVersion
    request: TorchTrainingRequest

    def __init__(
        self,
        user_id: int,
        experiment: Experiment,
        request: TorchTrainingRequest,
        mlflow_experiment_name: str,
    ):
        self.user_id = user_id
        self.experiment = experiment
        self.request = request
        self.mlflow_experiment_name = mlflow_experiment_name
        self.loggers: List[Logger] = []
        self._setup_loggers()

    def get_checkpoint_callback(self):
        """Gets the MonitoringCheckpoint passed to the trainer"""
        return self.checkpoint_callback

    def fit(self, **args):
        return fit(**args)

    def _setup_loggers(self):
        self.loggers = [
            MarinerLogger(
                self.experiment.id,
                experiment_name=self.experiment.experiment_name or "",
                user_id=self.user_id,
            ),
            MLFlowLogger(experiment_name=self.mlflow_experiment_name),
        ]
