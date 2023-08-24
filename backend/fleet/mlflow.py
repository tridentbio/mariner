"""
This package contains the MLflow integration for the project.

This module will replace :mod:`mariner.core.mlflow` in the future.
"""
import os
import pickle
from tempfile import mkdtemp
from typing import Any, Union

import mlflow
import torch.nn
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking.client import MlflowClient

from .utils.data import PreprocessingPipeline


def log_torch_models_and_create_version(
    model_name: str,
    best_model: torch.nn.Module,
    last_model: torch.nn.Module,
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

    run = mlflow.active_run()
    assert run is not None, "no active run"
    best_model_relative_path = "best"
    last_model_relative_path = "last"

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


def save_pipeline(pipeline: PreprocessingPipeline) -> None:
    """Save the pipeline as an artifact.

    Args:
        pipeline: pipeline object.
        run_id: run_id string of the training experiment.
    """
    tmp_dir = mkdtemp()
    artifact_path = os.path.join(tmp_dir, "pipeline.pkl")
    open(artifact_path, "wb").write(pickle.dumps(pipeline.get_state()))
    mlflow.log_artifact(artifact_path)


def load_pipeline(run_id) -> PreprocessingPipeline:
    """Load the pipeline from an artifact.

    Args:
        run_id: run_id string of the training experiment.

    Returns:
        PreprocessingPipeline: pipeline object.
    """
    tmp_dir = mkdtemp()
    artifact_path = os.path.join(tmp_dir, "pipeline.pkl")
    mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="pipeline.pkl", dst_path=tmp_dir
    )

    pipeline_state = pickle.load(open(artifact_path, "rb"))
    return PreprocessingPipeline.load_from_state(pipeline_state)
