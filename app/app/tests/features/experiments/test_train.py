import asyncio
import time

import pytest

from app.builder.dataset import DataModule
from app.features.dataset.model import Dataset
from app.features.experiments.schema import Experiment, TrainingRequest
from app.features.experiments.tasks import ExperimentManager, ExperimentView
from app.features.experiments.train.custom_logger import AppLogger
from app.features.experiments.train.run import start_training
from app.features.model.schema.model import Model
from app.tests.utils.utils import random_lower_string


@pytest.fixture(scope="function")
def task_manager():
    manager = ExperimentManager()
    yield manager
    for item in manager.experiments.values():
        item.task.cancel()


@pytest.mark.asyncio
async def test_add_task_remove_when_done(
    task_manager: ExperimentManager, some_experiment: Experiment
):
    async def sleep():
        time.sleep(3)
        return 42

    sleep_task = asyncio.create_task(sleep())
    logger = AppLogger(some_experiment.experiment_id, "teste", 1)
    task_manager.add_experiment(
        ExperimentView(task=sleep_task, experiment_id="1", user_id=1, logger=logger)
    )
    result = await sleep_task
    assert result == 42
    assert "1" not in task_manager.experiments


@pytest.mark.asyncio
async def test_start_training(
    some_dataset: Dataset,
    some_model: Model,
):
    version = some_model.versions[-1]
    exp_name = random_lower_string()
    request = TrainingRequest(
        epochs=20,
        learning_rate=1e-3,
        experiment_name=exp_name,
        model_name=some_model.name,
        model_version=version.model_version,
    )
    model = version.build_torch_model()
    featurizers_config = version.config.featurizers
    data_module = DataModule(
        featurizers_config=featurizers_config,
        data=some_dataset.get_dataframe(),
        dataset_config=version.config.dataset,
        split_target=some_dataset.split_target,
        split_type=some_dataset.split_type,
    )
    logger = AppLogger("1", "teste", 1)
    task = await start_training(model, request, data_module, loggers=logger)
    assert task
    assert logger

    await task
    #assert "train_loss" in logger.running_history
    #assert len(logger.running_history["train_loss"]) == request.epochs
