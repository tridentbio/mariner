import asyncio
from asyncio.tasks import Task

import logging
import ray
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer.trainer import Trainer

from mariner.schemas.experiment_schemas import TrainingRequest
from mariner.train.custom_logger import AppLogger
from model_builder.dataset import DataModule

if not ray.is_initialized():
    ray.init(address=settings.RAY_ADDRESS, logging_level=logging.ERROR)


async def train_run(
    trainer: Trainer,
    model: LightningModule,
    data_module: DataModule,
) -> LightningModule:
    trainer.fit(model, data_module.train_dataloader())
    return model


@ray.remote
def train_run_sync(
    trainer: Trainer,
    model: LightningModule,
    data_module: DataModule,
) -> LightningModule:
    trainer.fit(model, data_module.train_dataloader())
    return model


async def unwrap_ref(obj_ref: ray.ObjectRef) -> LightningModule:
    result = await obj_ref
    return result


async def start_training(
    model: LightningModule,
    training_request: TrainingRequest,
    data_module: DataModule,
    loggers: AppLogger,
) -> Task:
    data_module.setup(stage="fit")
    trainer = Trainer(
        max_epochs=training_request.epochs,
        logger=loggers,
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    module_ray_ref = train_run_sync.remote(trainer, model, data_module)
    coroutine = unwrap_ref(module_ray_ref)
    task = asyncio.create_task(coroutine)
    return task
