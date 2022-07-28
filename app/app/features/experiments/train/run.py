import asyncio
from asyncio.tasks import Task

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer.trainer import Trainer

from app.builder.dataset import DataModule
from app.features.experiments.schema import TrainingRequest
from app.features.experiments.train.custom_logger import AppLogger


async def train_run(
    trainer: Trainer,
    model: LightningModule,
    data_module: DataModule,
) -> LightningModule:
    trainer.fit(model, data_module.train_dataloader())
    return model


async def start_training(
    model: LightningModule,
    training_request: TrainingRequest,
    data_module: DataModule,
    loggers: AppLogger,
) -> Task:
    data_module.setup(stage="fit")
    trainer = Trainer(
        max_epochs=training_request.epochs, logger=loggers, log_every_n_steps=1
    )
    coroutine = train_run(trainer, model, data_module)
    task = asyncio.create_task(coroutine)
    return task
