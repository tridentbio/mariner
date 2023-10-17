"""
Mlflow Service
"""

import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import mlflow.tracking
from mlflow.exceptions import RestException

from fleet.torch_.models import CustomModel


class HandleRESTException:
    """
    Decorator to handle RESTException from mlflow.
    """

    def __init__(self, return_value=None):
        self.return_value = return_value

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RestException as exc:
                if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
                    return self.return_value
                raise

        return wrapper


def create_tracking_client():
    """Creates a mlflow tracking client MlflowClient"""
    client = mlflow.tracking.MlflowClient()
    return client


@HandleRESTException()
def get_registered_model(
    name: str, client: mlflow.tracking.MlflowClient | None = None
):
    """Gets the registered model with the given name.

    Args:
        name: Name of the registered model to get.
        client: mlflow tracking client.

    Returns:
        Registered model with the given name or None if the
        model is not found.
    """
    if not client:
        client = create_tracking_client()
    return client.get_registered_model(name)


@HandleRESTException()
def get_experiment(
    name: str | None,
    client: mlflow.tracking.MlflowClient | None = None,
):
    """Gets the experiment with the given name.

    Args:
        name: Name of the experiment to get.
        client: mlflow tracking client.
    """
    if not client:
        client = create_tracking_client()

    return client.get_experiment_by_name(name)


@HandleRESTException()
def get_run(run_id: str, client: mlflow.tracking.MlflowClient | None = None):
    """Gets the run with the given id.

    Args:
        run_id: Id of the run to get.

    Returns:
        Run with the given id.
    """
    if not client:
        client = create_tracking_client()
    run = client.get_run(run_id)
    return run


def search_experiments(
    experiment_name: str, client: mlflow.tracking.MlflowClient | None = None
):
    """Searches for experiments with the given name.

    Args:
        experiment_name: Name of the experiment to search for.

    Returns:
        List of experiments with the given name.
    """
    if not client:
        client = create_tracking_client()
    experiments = client.search_experiments(
        filter_string=f"name ILIKE '{experiment_name}%'",
        order_by=["creation_time DESC"],
    )
    return experiments


def create_registered_model(
    name: str,
    client: mlflow.tracking.MlflowClient | None = None,
    description: str | None = None,
    tags: dict[str, str] | None = None,
):
    """Creates a mlflow Model entity.

    Args:
        client: mlflow tracking client
        name: Name of the model
        description: Description of the model
        tags: List of tags of the project.
    """
    if not client:
        client = create_tracking_client()
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
