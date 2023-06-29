from typing import Optional

import mlflow.pyfunc
import mlflow.tracking
import pytest
import torch.nn
from mlflow.entities.model_registry.registered_model import \
    ModelVersion as MlflowModelVersion
from mlflow.entities.model_registry.registered_model import RegisteredModel
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository

from fleet.torch_.models import CustomModel
from tests.fixtures.model import mock_model
from tests.utils.utils import random_lower_string


def create_model_version(
    client: mlflow.tracking.MlflowClient,
    name: str,
    model: torch.nn.Module,
    artifact_path: Optional[str] = None,
    desc: Optional[str] = None,
) -> MlflowModelVersion:
    if not artifact_path:
        artifact_path = name
    active_run = mlflow.active_run()
    if active_run is not None:
        mlflow.end_run()
    with mlflow.start_run() as run:
        mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
        )
    runs_uri = f"runs:/{run.info.run_id}/{artifact_path}"
    model_src = RunsArtifactRepository.get_underlying_uri(runs_uri)
    version = client.create_model_version(
        name, model_src, run.info.run_id, description=desc
    )
    return version


@pytest.fixture()
def mlflow_model():
    client = mlflow.tracking.MlflowClient()
    model = client.create_registered_model(random_lower_string())
    model_config = mock_model().config
    file = CustomModel(
        config=model_config.spec, dataset_config=model_config.dataset
    )
    create_model_version(client, model.name, file)
    yield client.get_registered_model(model.name)
    client.delete_registered_model(model.name)


@pytest.mark.integration
def test_get_model(mlflow_model: RegisteredModel):
    assert mlflow_model.latest_versions
    version = mlflow_model.latest_versions[-1].version
    model = mlflow.pyfunc.load_model(f"models:/{mlflow_model.name}/{version}")
    assert isinstance(model, mlflow.pyfunc.PyFuncModel)
