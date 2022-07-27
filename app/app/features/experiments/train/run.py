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
from app.features.experiments.tasks import get_exp_manager
from app.features.experiments.train.custom_logger import AppLogger
from app.features.model.builder import CustomDataset


async def train_run(
    experiment: Experiment,
    model: LightningModule,
    dataloader: DataLoader,
    training_request: TrainingRequest,
):
    client = MlflowClient()
    run = client.create_run(experiment.experiment_id)
    logger = MLFlowLogger(experiment_name=experiment.name, run_id=run.info.run_id)
    applogger = AppLogger()
    loggers = [logger, applogger]
    trainer = Trainer(max_epochs=training_request.epochs, logger=loggers)
    trainer.fit(model, dataloader)
    mlflow.pytorch.log_model(model, get_artifact_uri(run.info.run_id))


async def start_training(
    model: LightningModule, training_request: TrainingRequest, dataset: CustomDataset
) -> tuple[str, Task]:
    # TODO: Customize learning rate, preferably here
    mlflow.create_experiment(training_request.experiment_name)
    experiment = mlflow.get_experiment_by_name(training_request.experiment_name)
    assert experiment
    dataloader = DataLoader(dataset)
    coroutine = train_run(experiment, model, dataloader, training_request)
    task = asyncio.create_task(coroutine)

    return experiment.experiment_id, task
