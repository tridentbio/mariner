import io
from typing import Optional, Union

import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import mlflow.tracking
from fastapi.datastructures import UploadFile
from mlflow.deployments import get_deploy_client
from mlflow.deployments.base import BaseDeploymentClient
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.entities.model_registry.registered_model import RegisteredModel
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from ray import serve

from app.features.model.schema.model import Model
from app.tests.data.torch_target_model import ExampleModel


def create_model_version(
    client: mlflow.tracking.MlflowClient,
    name: str,
    file: UploadFile,
    artifact_path: Optional[str] = None,
    desc: Optional[str] = None,
) -> ModelVersion:
    file = io.BytesIO(file.file.read())
    model = ExampleModel()
    if not artifact_path:
        artifact_path = name
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


def create_tracking_client():
    return mlflow.tracking.MlflowClient()


def create_registered_model(
    client: mlflow.tracking.MlflowClient,
    name: str,
    torchscript: Optional[UploadFile] = None,
    tags: Optional[dict[str, str]] = None,
    description: Optional[str] = None,
    version_description: Optional[str] = None,
) -> tuple[RegisteredModel, Optional[ModelVersion]]:
    registered_model = client.create_registered_model(
        name, tags=tags, description=description
    )
    if not torchscript:
        return registered_model, None
    version = create_model_version(client, name, torchscript, desc=version_description)
    return registered_model, version


def get_deployment_plugin() -> BaseDeploymentClient:
    client = get_deploy_client("ray-serve://ray-head:10001")
    assert client is not None
    return client


def create_deployment_with_endpoint(deployment_name: str, model_uri: str):
    # ray.init(address=f'ray://ray-head:10001')
    serve.start(detached=True)
    ray_plugin = get_deployment_plugin()
    deployment = ray_plugin.create_deployment(
        name=deployment_name,
        model_uri=model_uri,
        # config={"num_replicas": 1}
    )
    print(deployment)
    return deployment


def get_registry_model(model_registry_name: str):
    client = create_tracking_client()
    registered_model = client.get_registered_model(model_registry_name)
    return registered_model


def get_model(model: Model, version: Optional[Union[int, str]]):
    mlflowmodel = mlflow.pyfunc.load_model(model.get_model_uri(version))
    return mlflowmodel
