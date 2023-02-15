from typing import Optional, Union

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


def log_models_and_create_version(
    model_name: str,
    best_model: torch.nn.Module,
    last_model: torch.nn.Module,
    run_id: str,
    version_description: Optional[str] = None,
    client: Optional[mlflow.tracking.MlflowClient] = None,
) -> ModelVersion:
    """Use mlflow API to log the trained models and create a registry model version

    Defines the mlflow after trainining handler

    Args:
        model_name: name of the registered model
        best_model: best torch model
        last_model: last torch model
        run_id: run_id string of the training experiment
        version_description: version description
        client: Optional mlflow client

    Returns:
        ModelVersion: The mlflow ModelVersion created
    """
    best_model_relative_path = "best"
    last_model_relative_path = "last"

    with mlflow.start_run(run_id=run_id) as run:
        mlflow.pytorch.log_model(
            best_model,
            artifact_path=best_model_relative_path,
        )
        mlflow.pytorch.log_model(
            last_model,
            artifact_path=last_model_relative_path,
        )
    # log model version as best model
    runs_uri = f"runs:/{run.info.run_id}/{best_model_relative_path}"
    # gets underlying s3 path of the run artifact
    model_src = RunsArtifactRepository.get_underlying_uri(runs_uri)
    if not client:
        client = MlflowClient()
    version = client.create_model_version(
        model_name, model_src, run.info.run_id, description=version_description
    )
    return version


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
    description: Union[str, None] = None,
    tags: Union[dict[str, str], None] = None,
):
    registered_model = client.create_registered_model(
        name, tags=tags, description=description
    )
    return registered_model


def create_registered_model_and_version(
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
