"""
Mlflow Service
"""
from typing import Union

import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import mlflow.tracking

from fleet.torch_.models import CustomModel


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


def get_model_by_uri(model_uri: str, map_location=None) -> CustomModel:
    """The uri specifying the model.

    Args:
        model_uri: URI referring to the ML model directory.

    Returns:
        torch instance of the model.
    """
    mlflowmodel = mlflow.pytorch.load_model(
        model_uri, map_location=map_location
    )
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
