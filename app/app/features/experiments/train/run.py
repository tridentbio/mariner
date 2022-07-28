import asyncio
from asyncio.tasks import Task

import mlflow
from mlflow.entities.experiment import Experiment
from mlflow.tracking.artifact_utils import get_artifact_uri
from mlflow.tracking.client import MlflowClient
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers.mlflow import MLFlowLogger
from pytorch_lightning.trainer.trainer import Trainer
from torch_geometric.loader.dataloader import DataLoader

from app.features.experiments.schema import TrainingRequest
from app.features.experiments.train.custom_logger import AppLogger
from app.builder.dataset import CustomDataset, DataModule


async def train_run(
    trainer: Trainer,
    experiment: Experiment,
    model: LightningModule,
    data_module: DataModule,
):
    client = MlflowClient()
    run = client.create_run(experiment.experiment_id)
    trainer.fit(model, data_module.train_dataloader())
    mlflow.pytorch.log_model(model, get_artifact_uri(run.info.run_id))


async def start_training(
    model: LightningModule, training_request: TrainingRequest, data_module: DataModule
) -> tuple[str, Task, AppLogger]:
    applogger = AppLogger()
    trainer = Trainer(max_epochs=training_request.epochs, logger=applogger)
    mlflow.create_experiment(training_request.experiment_name)
    experiment = mlflow.get_experiment_by_name(training_request.experiment_name)
    assert experiment
    coroutine = train_run(trainer, experiment, model, data_module, training_request)
    task = asyncio.create_task(coroutine)
    return experiment.experiment_id, task, applogger
