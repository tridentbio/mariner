import asyncio
import time
import pytest
import mlflow
from app.features.dataset.model import Dataset
from app.features.experiments.schema import TrainingRequest
from app.features.model.builder import CustomDataset
from app.features.model.schema.model import Model
from app.features.experiments.train.run import start_training
from app.features.experiments.tasks import ExperimentManager, get_exp_manager
from app.tests.utils.utils import random_lower_string


@pytest.fixture(scope="function")
def task_manager():
    manager = ExperimentManager()
    yield manager
    for item in manager.tasks.values():
        item.cancel()

@pytest.mark.asyncio
async def test_add_task_remove_when_done(task_manager: ExperimentManager):
    async def sleep():
        time.sleep(3)
        return 42
    sleep_task = asyncio.create_task(sleep())
    task_manager.add_experiment('1', sleep_task)
    result = await sleep_task
    assert result == 42
    assert '1' not in task_manager.tasks



@pytest.mark.asyncio
async def test_start_training(
    some_dataset: Dataset,
    some_model: Model,
):
    experiment_manager = get_exp_manager()
    version = some_model.versions[-1]
    exp_name = random_lower_string()
    request = TrainingRequest(
        epochs=1,
        learning_rate=1e-3,
        experiment_name=exp_name,
        model_name=some_model.name,
        model_version=version.model_version
    )
    model = version.build_torch_model()
    df = some_dataset.get_dataframe()
    dataset = CustomDataset(df, version.config)
    experiment_id = await start_training(model, request, dataset)
    assert experiment_id
    experiement = mlflow.get_experiment(experiment_id)
    assert experiement.name == exp_name
    task = experiment_manager.get_task(experiment_id)
    assert task
    
    ## TODO: await for task completion and check proper outcome


