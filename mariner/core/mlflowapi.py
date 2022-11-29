from typing import Optional

import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import mlflow.tracking
import ray
import torch
from mlflow.deployments import get_deploy_client
from mlflow.deployments.base import BaseDeploymentClient
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.entities.model_registry.registered_model import RegisteredModel
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking.client import MlflowClient


def create_model_version(
    client: mlflow.tracking.MlflowClient,
    name: str,
    model: torch.nn.Module,
    artifact_path: Optional[str] = None,
    desc: Optional[str] = None,
) -> ModelVersion:
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


def create_tracking_client():
    client = mlflow.tracking.MlflowClient()
    return client


def create_registered_model(
    client: mlflow.tracking.MlflowClient,
    name: str,
    torchscript: Optional[torch.nn.Module] = None,
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


_client = None


def get_deployment_plugin() -> BaseDeploymentClient:
    global _client
    if _client is None:
        _client = get_deploy_client("ray-serve://ray-head:10001")
        assert _client is not None
    return _client


def create_deployment_with_endpoint(deployment_name: str, model_uri: str):
    if ray.is_initialized():
        ray.shutdown()
    ray_plugin = get_deployment_plugin()
    deployment = ray_plugin.create_deployment(
        name=deployment_name,
        model_uri=model_uri,
    )
    return deployment


def get_registry_model(model_registry_name: str, client: Optional[MlflowClient] = None):
    if not client:
        client = create_tracking_client()
    registered_model = client.get_registered_model(model_registry_name)
    return registered_model


def get_model_by_uri(model_uri: str):
    mlflowmodel = mlflow.pytorch.load_model(model_uri)
    return mlflowmodel


def get_model_version(model_name: str, version: str):
    client = create_tracking_client()
    mlflowversion = client.get_model_version(model_name, version)
    return mlflowversion
