from typing import List

import ray
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.loggers.mlflow import MLFlowLogger
from pytorch_lightning.trainer.trainer import Trainer

from mariner.entities.experiment import Experiment
from mariner.schemas.dataset_schemas import Dataset
from mariner.schemas.experiment_schemas import TrainingRequest
from mariner.schemas.model_schemas import ModelVersion
from mariner.train.custom_logger import AppLogger
from model_builder.dataset import DataModule
from model_builder.model import CustomModel


@ray.remote
class TrainingActor:

    checkpoint_callback: ModelCheckpoint
    dataset: Dataset
    modelversion: ModelVersion
    requeset: TrainingRequest

    def __init__(
        self, dataset: Dataset, modelversion: ModelVersion, request: TrainingRequest
    ):
        self.dataset = dataset
        self.modelversion = modelversion
        self.request = request
        self.loggers: List[Logger] = []

    def get_checkpoint_callback(self):
        return self.checkpoint_callback

    def setup_loggers(
        self, mariner_experiment: Experiment, mlflow_experiment_name: str, user_id: int
    ):
        self.loggers = [
            AppLogger(
                mariner_experiment.id,
                experiment_name=mariner_experiment.experiment_name,
                user_id=user_id,
            ),
            MLFlowLogger(experiment_name=mlflow_experiment_name),
        ]

    def setup_callbacks(
        self,
    ):
        monitoring_config = self.request.checkpoint_config
        self.checkpoint_callback = ModelCheckpoint(
            monitor=monitoring_config.metric_key,
            mode=monitoring_config.mode,
            save_last=True,
        )

    def train(self):
        modelconfig = self.modelversion.config
        model = CustomModel(config=modelconfig)
        df = self.dataset.get_dataframe()
        datamodule = DataModule(
            featurizers_config=modelconfig.featurizers,
            data=df,
            dataset_config=modelconfig.dataset,
            split_target=self.dataset.split_target,
            split_type=self.dataset.split_type,
            batch_size=self.request.batch_size or 32,
        )
        trainer = Trainer(
            max_epochs=self.request.epochs,
            logger=self.loggers,
            log_every_n_steps=1,
            enable_progress_bar=False,
            callbacks=[self.checkpoint_callback],
        )
        datamodule.setup()
        trainer.fit(model, datamodule)
        return model
