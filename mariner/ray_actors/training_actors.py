"""
Actors for Training, validation and testing models
"""
from typing import List, Union

import ray
from mlflow.tracking import MlflowClient
from pytorch_lightning import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.loggers.mlflow import MLFlowLogger
from pytorch_lightning.trainer.trainer import Trainer

from mariner.core.mlflowapi import log_models_and_create_version
from mariner.schemas.dataset_schemas import Dataset
from mariner.schemas.experiment_schemas import Experiment, TrainingRequest
from mariner.schemas.model_schemas import ModelVersion
from mariner.train.custom_logger import AppLogger
from model_builder.dataset import DataModule
from model_builder.model import CustomModel


@ray.remote
class TrainingActor:
    """Runs training with needed mlflow and custom logging"""

    checkpoint_callback: ModelCheckpoint
    early_stopping_callback: Union[EarlyStopping, None] = None
    dataset: Dataset
    modelversion: ModelVersion
    request: TrainingRequest

    def __init__(
        self,
        user_id: int,
        experiment: Experiment,
        request: TrainingRequest,
        mlflow_experiment_name: str,
    ):
        self.user_id = user_id
        self.experiment = experiment
        self.request = request
        self.mlflow_experiment_name = mlflow_experiment_name
        self.loggers: List[Logger] = []
        self._setup_loggers()
        self._setup_callbacks()

    def get_checkpoint_callback(self):
        """Gets the MonitoringCheckpoint passed to the trainer"""
        return self.checkpoint_callback

    def train(self, dataset: Dataset):
        """Performs the training experiment on the dataset, log
        best and last models and return the best model version mlflow
        data

        Args:
            dataset: dataset to perform the experiment
        """
        modelconfig = self.experiment.model_version.config
        model = CustomModel(config=modelconfig)
        model.set_optimizer(self.request.optimizer)
        df = dataset.get_dataframe()
        batch_size = self.request.batch_size or 32
        datamodule = DataModule(
            config=modelconfig,
            data=df,
            split_target=dataset.split_target,
            split_type=dataset.split_type,
            batch_size=batch_size,
        )
        callbacks: List[Callback] = [self.checkpoint_callback]
        if self.early_stopping_callback is not None:
            callbacks.append(self.early_stopping_callback)
        trainer = Trainer(
            max_epochs=self.request.epochs,
            logger=self.loggers,
            log_every_n_steps=max(batch_size // 2, 10),
            enable_progress_bar=False,
            callbacks=[
                callback
                for callback in (self.checkpoint_callback, self.early_stopping_callback)
                if callback
            ],
        )
        datamodule.setup()
        trainer.fit(model, datamodule)
        return self._log_models()

    def _log_models(self):
        """Logs best and last models to mlflow tracker, and return the model version
        that performed best on the monitored metric
        """
        client = MlflowClient()
        best_model_path = self.checkpoint_callback.best_model_path
        best_model = CustomModel.load_from_checkpoint(best_model_path)
        assert isinstance(best_model, CustomModel), "best_model failed to be logged"
        last_model_path = self.checkpoint_callback.last_model_path
        last_model = CustomModel.load_from_checkpoint(last_model_path)
        assert isinstance(last_model, CustomModel), "last_model failed to be logged"
        runs = client.search_runs(experiment_ids=[self.experiment.mlflow_id])
        assert len(runs) == 1
        run = runs[0]
        run_id = run.info.run_id
        mlflow_model_version = log_models_and_create_version(
            self.experiment.model_version.mlflow_model_name,
            best_model,
            last_model,
            run_id,
            client=client,
        )
        return mlflow_model_version

    def _setup_loggers(self):
        self.loggers = [
            AppLogger(
                self.experiment.id,
                experiment_name=self.experiment.experiment_name or "",
                user_id=self.user_id,
            ),
            MLFlowLogger(experiment_name=self.mlflow_experiment_name),
        ]

    def _setup_callbacks(
        self,
    ):
        monitoring_config = self.request.checkpoint_config
        early_stopping_config = self.request.early_stopping_config
        self.checkpoint_callback = ModelCheckpoint(
            monitor=monitoring_config.metric_key,
            mode=monitoring_config.mode,
            save_last=True,
        )
        if early_stopping_config:
            self.early_stopping_callback = EarlyStopping(
                monitor=early_stopping_config.metric_key,
                mode=early_stopping_config.mode,
                min_delta=early_stopping_config.min_delta,
                patience=early_stopping_config.patience,
                check_finite=early_stopping_config.check_finite,
            )
