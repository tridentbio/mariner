import asyncio
import logging
from asyncio.tasks import Task
from typing import List, Union

import ray
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.trainer.trainer import Trainer

from mariner.core.config import settings
from mariner.schemas.experiment_schemas import TrainingRequest
from model_builder.dataset import DataModule


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
    loggers: Union[List[LightningLoggerBase], LightningLoggerBase],
) -> Task:
    data_module.setup(stage="fit")
    trainer = Trainer(
        max_epochs=training_request.epochs,
        logger=loggers,
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        progress_bar_refresh_rate=0,
    )
    conn_ray = ray.init(
        address=settings.RAY_ADDRESS, logging_level=logging.ERROR, allow_multiple=True
    )
    with conn_ray:
        module_ray_ref = train_run_sync.remote(trainer, model, data_module)
    coroutine = unwrap_ref(module_ray_ref)
    task = asyncio.create_task(coroutine)
    return task
