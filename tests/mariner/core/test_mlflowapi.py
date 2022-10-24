import mlflow.pyfunc
import mlflow.tracking
import pytest
from mlflow.entities.model_registry.registered_model import RegisteredModel

from mariner.core.mlflowapi import (
    create_deployment_with_endpoint,
    create_model_version,
)
from model_builder.model import CustomModel
from tests.conftest import mock_model
from tests.utils.utils import random_lower_string


@pytest.fixture()
def mlflow_model():
    client = mlflow.tracking.MlflowClient()
    model = client.create_registered_model(random_lower_string())
    model_config = mock_model().config
    file = CustomModel(model_config)
    create_model_version(client, model.name, file)
    yield client.get_registered_model(model.name)
    client.delete_registered_model(model.name)


@pytest.mark.long
def test_create_deployment(mlflow_model: RegisteredModel):
    assert mlflow_model.latest_versions
    version = mlflow_model.latest_versions[-1].version
    model_uri = f"models:/{mlflow_model.name}/{version}"
    deployment_name = random_lower_string()
    deployment = create_deployment_with_endpoint(deployment_name, model_uri)
    assert deployment is not None


def test_get_model(mlflow_model: RegisteredModel):
    assert mlflow_model.latest_versions
    version = mlflow_model.latest_versions[-1].version
    model = mlflow.pyfunc.load_model(f"models:/{mlflow_model.name}/{version}")
    assert isinstance(model, mlflow.pyfunc.PyFuncModel)
