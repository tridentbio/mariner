"""
The fleet.models module provides a high-level interface for working with various
machine learning algorithms, including those from popular libraries like torch 
and scikit-learn. These algorithms are designed to tackle a range regression and
classification problems on chemical and life science dataset.
"""


import logging
from dataclasses import dataclass
from typing import List, Union

import mlflow
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from mlflow.entities.model_registry.model_version import ModelVersion
from pandas import DataFrame
from pydantic import BaseModel

from fleet.base_schemas import (
    BaseModelFunctions,
    FleetModelSpec,
    TorchModelSpec,
)
from fleet.scikit_.model_functions import SciKitFunctions
from fleet.scikit_.schemas import SklearnModelSpec
from fleet.torch_.model_functions import TorchFunctions
from fleet.torch_.schemas import TorchTrainingConfig
from fleet.train.custom_logger import MarinerLogger
from mariner.core.aws import download_file_as_dataframe

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


@dataclass
class Result:
    """
    Holds properties created during the :py:func:`fit`.
    """

    mlflow_experiment_id: str
    mlflow_model_version: Union[None, ModelVersion]


def fit(
    spec: FleetModelSpec,
    train_config: BaseModel,  # todo: make this type narrower
    mlflow_model_name: str,
    mlflow_experiment_name: str,
    experiment_id: Union[None, int] = None,
    experiment_name: Union[None, str] = None,
    user_id: Union[None, int] = None,
    datamodule_args: Union[None, dict] = None,
    dataset_uri: Union[None, str] = None,
    dataset: Union[None, DataFrame] = None,
):
    """
    Fits a model to a dataset.

    Depending on the spec's class, different machine learning frameworks are used.

    Args:
        spec: the model specification to be trained.
        train_config: configuration train
        mlflow_model_name: model string identifier used to publish the mlflow model name.
        mlflow_experiment_name: mlflow experiment string identifier.
        experiment_id: id of the :py:class:mariner.entities.experiment.Experiment` entity
        experiment_name: name of the :py:class:`mariner.entities.experiment.Experiment`
        user_id: id of the :py:class:`mariner.entities.user.User`.
        datamodule_args: the arguments for the :py:class:`DataModule` used when training
            torch models.
        dataset_uri: S3 URI to download the dataset.
        dataset: A :py:class:`DataFrame`

    Raises:
        RuntimeError: When the experiment creation in mlflow fails.
        ValueError: dataset_uri
    """
    functions: Union[BaseModelFunctions, None] = None
    try:
        mlflow_experiment_id = mlflow.create_experiment(mlflow_experiment_name)
    except Exception as exp:
        raise RuntimeError("Failed to create mlflow experiment") from exp

    if dataset is None and dataset_uri is None:
        raise ValueError("dataset_uri or dataset must be passed to fit()")
    elif dataset is None and dataset_uri is not None:
        assert dataset_uri.startswith(
            "s3://"
        ), "dataset_uri is invalid: should start with s3://"
        # Splits s3 uri into bucket and object key
        bucket, key = dataset_uri[5:].split("/", 1)
        dataset = download_file_as_dataframe(bucket, key)
    assert isinstance(dataset, DataFrame), "Dataset must be DataFrame"

    if isinstance(spec, TorchModelSpec):
        functions = TorchFunctions()
        loggers: List[Logger] = [MLFlowLogger(experiment_name=mlflow_experiment_name)]
        if (
            experiment_id is not None
            and experiment_name is not None
            and user_id is not None
        ):
            loggers.append(
                MarinerLogger(
                    experiment_id=experiment_id,
                    experiment_name=experiment_name,
                    user_id=user_id,
                )
            )
        else:
            LOG.warning(
                "Not creating MarinerLogger because experiment_id or experiment_name or user_id are missing"
            )

        functions.loggers = loggers
        assert isinstance(
            train_config, TorchTrainingConfig
        ), f"train_config should be TorchTrainingConfig but is {train_config.__class__}"
        functions.train(
            spec=spec,
            dataset=dataset,
            params=train_config,
            datamodule_args=datamodule_args if datamodule_args else {},
        )
    elif isinstance(spec, SklearnModelSpec):
        functions = SciKitFunctions(spec, dataset)
        with mlflow.start_run(nested=True, experiment_id=mlflow_experiment_id) as run:
            functions.train()
            functions.log_model(
                model_name=mlflow_model_name,
                run_id=run.info.run_id,
            )
            functions.val()
            functions.test()

    if not functions:
        raise ValueError("Can't find functions for spec")

    mlflow_model_version = functions.log_models(
        mlflow_model_name=mlflow_model_name,
        mlflow_experiment_id=mlflow_experiment_id,
    )

    return Result(
        mlflow_experiment_id=mlflow_experiment_id,
        mlflow_model_version=mlflow_model_version,
    )
