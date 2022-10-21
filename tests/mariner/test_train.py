import asyncio
import time

import pytest
from sqlalchemy.orm import Session
from mariner.schemas.dataset_schemas import Dataset
from mariner.schemas.experiment_schemas import Experiment, TrainingRequest
from mariner.schemas.model_schemas import Model

from mariner.tasks import ExperimentManager, ExperimentView
from mariner.train.custom_logger import AppLogger
from mariner.train.run import start_training
from model_builder.dataset import DataModule
from tests.conftest import get_test_user
from tests.utils.utils import random_lower_string


@pytest.fixture(scope="function")
def task_manager():
    manager = ExperimentManager()
    yield manager
    for item in manager.experiments.values():
        item.task.cancel()


@pytest.mark.asyncio
async def test_add_task_remove_when_done(
    db: Session, task_manager: ExperimentManager, some_experiment: Experiment
):
    async def sleep():
        time.sleep(3)
        return 42

    sleep_task = asyncio.create_task(sleep())
    user = get_test_user(db)
    task_manager.add_experiment(
        ExperimentView(
            task=sleep_task, experiment_id=some_experiment.id, user_id=user.id
        )
    )
    result = await sleep_task
    assert result == 42
    assert "1" not in task_manager.experiments


@pytest.mark.asyncio
@pytest.mark.long
async def test_start_training(
    db: Session,
    some_dataset: Dataset,
    some_model: Model,
    some_experiment: Experiment,
):
    version = some_model.versions[-1]
    exp_name = random_lower_string()
    request = TrainingRequest(
        epochs=5,
        learning_rate=1e-3,
        name=exp_name,
        model_version_id=version.id,
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
    user = get_test_user(db)
    logger = AppLogger(some_experiment.id, "teste", user.id)
    task = await start_training(model, request, data_module, loggers=logger)
    assert task
    assert logger

    await task
