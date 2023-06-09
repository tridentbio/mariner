"""
Mlflow Service
"""
from typing import Optional, Union, Any

import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import mlflow.tracking
import torch
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking.client import MlflowClient

from fleet.torch_.models import CustomModel


def log_torch_models_and_create_version(
    model_name: str,
    best_model: torch.nn.Module,
    last_model: torch.nn.Module,
    run_id: str,
    version_description: Optional[str] = None,
    client: Optional[mlflow.tracking.MlflowClient] = None,
) -> ModelVersion:
    """Use mlflow API to log the trained models and create a registry model version.

    Args:
        model_name: name of the registered model.
        best_model: best torch model.
        last_model: last torch model.
        run_id: run_id string of the training experiment.
        version_description: version description.
        client: Optional mlflow client.

    Returns:
        ModelVersion: The mlflow ModelVersion created.
    """
    best_model_relative_path = "best"
    last_model_relative_path = "last"

    with mlflow.start_run(run_id=run_id, nested=True) as run:
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


def log_sklearn_model_and_create_version(
    model: Any,
    model_name: Union[None, str] = None,
    run_id: Union[None, str] = None,
    version_description: Optional[str] = None,
    client: Optional[mlflow.tracking.MlflowClient] = None,
) -> Union[None, ModelVersion]:
    """Use mlflow API to log the trained model and create a registry model version.

    Args:
        model_name: name of the registered model.
        model: sklearn model.
        run_id: run_id string of the training experiment.
        version_description: version description.
    """
    with mlflow.start_run(run_id=run_id, nested=True) as run:  # type: ignore
        mlflow.sklearn.log_model(model, artifact_path="model")
    runs_uri = f"runs:/{run.info.run_id}/model"
    # gets underlying s3 path of the run artifact
    model_src = RunsArtifactRepository.get_underlying_uri(runs_uri)
    if not client:
        client = MlflowClient()
    if model_name:
        version = client.create_model_version(
            model_name, model_src, run.info.run_id, description=version_description
        )
        return version


def create_tracking_client():
    """Creates a mlflow tracking client MlflowClient"""
    client = mlflow.tracking.MlflowClient()
    return client


def create_registered_model(
    client: mlflow.tracking.MlflowClient,
    name: str,
    description: Union[str, None] = None,
    tags: Union[dict[str, str], None] = None,
):
    """Creates a mlflow Model entity.

    Args:
        client: mlflow tracking client
        name: Name of the model
        description: Description of the model
        tags: List of tags of the project.
    """
    registered_model = client.create_registered_model(
        name, tags=tags, description=description
    )
    return registered_model


def get_model_by_uri(model_uri: str) -> CustomModel:
    """The uri specifying the model.

    Args:
        model_uri: URI referring to the ML model directory.

    Returns:
        torch instance of the model.
    """
    mlflowmodel = mlflow.pytorch.load_model(model_uri)
    return mlflowmodel


def get_model_version(model_name: str, version: str):
    """Gets the model version information from the registry.

    Args:
        model_name: Name of the model>
        version: Name of the model version.
    """
    client = create_tracking_client()
    mlflowversion = client.get_model_version(model_name, version)
    return mlflowversion
