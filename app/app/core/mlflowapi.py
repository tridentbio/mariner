from typing import Optional

import mlflow
import mlflow.pytorch
import mlflow.tracking
import torch.jit
from fastapi.datastructures import UploadFile
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.entities.model_registry.registered_model import RegisteredModel
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository


def create_model_version(
    client: mlflow.tracking.MlflowClient,
    name: str,
    file: UploadFile,
    artifact_path: Optional[str] = None,
    desc: Optional[str] = None,
) -> ModelVersion:
    model = torch.jit.load(file)
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
    if torchscript is None:
        return registered_model, None
    version = create_model_version(client, name, torchscript, desc=version_description)
    return registered_model, version
