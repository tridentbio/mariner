"""
This package contains the MLflow integration for the project.

This module will replace :mod:`mariner.core.mlflow` in the future.
"""
from typing import Any, Union

import mlflow
import torch.nn
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking.client import MlflowClient


def log_torch_models_and_create_version(
    model_name: str,
    best_model: torch.nn.Module,
    last_model: torch.nn.Module,
    run_id: str,
    version_description: Union[None, str] = None,
    client: Union[None, mlflow.tracking.MlflowClient] = None,
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
    version_description: Union[None, str] = None,
    client: Union[None, mlflow.tracking.MlflowClient] = None,
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
            model_name,
            model_src,
            run.info.run_id,
            description=version_description,
        )
        return version
