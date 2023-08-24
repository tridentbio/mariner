"""
The fleet.models module provides a high-level interface for working with various
machine learning algorithms from popular libraries like torch 
and scikit-learn. These algorithms are designed to tackle a range regression and
classification problems on chemical and life science dataset.
"""


import logging
from dataclasses import dataclass
from typing import List, Union

import mlflow
import numpy as np
import pandas as pd
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
from fleet.dataset_schemas import (
    DatasetConfigWithPreprocessing,
    TorchDatasetConfig,
)
from fleet.mlflow import load_pipeline, save_pipeline
from fleet.scikit_.model_functions import SciKitFunctions
from fleet.scikit_.schemas import SklearnModelSpec
from fleet.torch_.model_functions import TorchFunctions
from fleet.torch_.schemas import TorchTrainingConfig
from fleet.train.custom_logger import MarinerLogger
from fleet.utils.dataset import (
    check_dataframe_conforms_dataset,
    converts_file_to_dataframe,
)
from mariner.core.aws import download_s3
from mariner.exceptions import InvalidDataframe

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


@dataclass
class Result:
    """
    Holds properties created during the :py:func:`fit`.
    """

    mlflow_experiment_id: str
    mlflow_model_version: Union[None, ModelVersion]


def _get_dataframe(
    dataset: Union[None, DataFrame], dataset_uri: Union[None, str]
) -> DataFrame:
    if dataset is None and dataset_uri is None:
        raise ValueError("dataset_uri or dataset must be passed to fit()")
    elif dataset is None and dataset_uri is not None:
        assert dataset_uri.startswith(
            "s3://"
        ), "dataset_uri is invalid: should start with s3://"
        # Splits s3 uri into bucket and object key
        bucket, key = dataset_uri[5:].split("/", 1)
        file = download_s3(key, bucket)
        return converts_file_to_dataframe(file)
    else:
        assert dataset is not None, "dataset_uri or dataset are required"
        return dataset


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
    if isinstance(spec, SklearnModelSpec) and isinstance(
        spec.dataset, DatasetConfigWithPreprocessing
    ):
        spec.dataset = spec.dataset.to_dataset_config()
    functions: Union[BaseModelFunctions, None] = None
    try:
        mlflow_experiment_id = mlflow.create_experiment(mlflow_experiment_name)
    except Exception as exc:
        LOG.exception(exc)
        raise RuntimeError("Failed to create mlflow experiment") from exc
    dataset = _get_dataframe(dataset, dataset_uri)
    assert isinstance(dataset, DataFrame), "Dataset must be DataFrame"

    with mlflow.start_run(
        nested=True, experiment_id=mlflow_experiment_id
    ) as run:
        if isinstance(spec, TorchModelSpec):
            functions = TorchFunctions(
                spec=spec,
                dataset=dataset,
            )
            loggers: List[Logger] = [
                MLFlowLogger(
                    run_id=run.info.run_id,
                    experiment_name=mlflow_experiment_name,
                )
            ]
            if (
                experiment_id is not None
                and experiment_name is not None
                and user_id is not None
            ):
                logger = MarinerLogger(
                    experiment_id=experiment_id,
                    experiment_name=experiment_name,
                    user_id=user_id,
                )
                loggers.append(logger)
            else:
                LOG.warning(
                    "Not creating MarinerLogger because experiment_id or experiment_name or user_id are missing"
                )

            functions.loggers = loggers
            assert isinstance(
                train_config, TorchTrainingConfig
            ), f"train_config should be TorchTrainingConfig but is {train_config.__class__}"
            functions.train(
                params=train_config,
                datamodule_args=datamodule_args if datamodule_args else {},
            )
            mlflow_model_version = functions.log_models(
                mlflow_model_name=mlflow_model_name, run_id=run.info.run_id
            )

        elif isinstance(spec, SklearnModelSpec):
            functions = SciKitFunctions(spec, dataset)
            functions.train()
            mlflow_model_version = functions.log_models(
                mlflow_model_name=mlflow_model_name,
                run_id=run.info.run_id,
            )
            functions.val()
        else:
            raise ValueError("Can't find functions for spec")

        save_pipeline(functions.preprocessing_pipeline)

    return Result(
        mlflow_experiment_id=mlflow_experiment_id,
        mlflow_model_version=mlflow_model_version,
    )


def run_test(
    spec: FleetModelSpec,
    mlflow_model_name: str,
    mlflow_model_version: str,
    dataset_uri: Union[None, str] = None,
    dataset: Union[None, DataFrame] = None,
):
    """Tests a model from any of the supported frameworks.

    Metrics are logged in the same mlflow run as the training.

    Args:
        spec: The model and dataset specifications.
        mlflow_model_name: The name of the registered model in mlflow.
        mlflow_model_version: The version of the model.
        dataset_uri(optional): The S3 dataset uri.
        dataset(optional): The dataframe with the dataset. Should have a step
            column.
    """
    if isinstance(spec, SklearnModelSpec) and isinstance(
        spec.dataset, DatasetConfigWithPreprocessing
    ):
        spec.dataset = spec.dataset.to_dataset_config()
    dataset = _get_dataframe(dataset, dataset_uri)
    client = mlflow.MlflowClient()
    model_uri = f"models:/{mlflow_model_name}/{mlflow_model_version}"
    modelversion = client.get_model_version(
        mlflow_model_name, mlflow_model_version
    )
    assert modelversion.run_id, (
        f"missing run_id model with version {mlflow_model_version} at model",
        f"{mlflow_model_name} at Mlflow",
    )
    if isinstance(spec, SklearnModelSpec):
        model = mlflow.sklearn.load_model(model_uri)
        pipeline = load_pipeline(modelversion.run_id)
        functions = SciKitFunctions(
            spec, dataset, model=model, preprocessing_pipeline=pipeline
        )
        with mlflow.start_run(nested=True, run_id=modelversion.run_id):
            functions.test()
    elif isinstance(spec, TorchModelSpec):
        # TODO:
        pass


def check_input(input_: pd.DataFrame, config: TorchDatasetConfig):
    """Checks if the input conforms to the dataset config.

    Args:
        input_: The dataframe with the input data.
        config: The dataset config.

    Raises:
        InvalidDataframe: When the input does not conform to the dataset config.
    """
    broken_checks = check_dataframe_conforms_dataset(input_, config)
    if len(broken_checks) > 0:
        raise InvalidDataframe(
            f"dataframe failed {len(broken_checks)} checks",
            reasons=[
                f"{col_name}: {rule}" for col_name, rule in broken_checks
            ],
        )


def predict(
    spec: FleetModelSpec,
    mlflow_model_name: str,
    mlflow_model_version: str,
    input_: Union[pd.DataFrame, dict],
):
    """Predicts with a model from any of the supported frameworks.

    Args:
        spec: The model and dataset specifications.
        mlflow_model_name: The name of the registered model in mlflow.
        mlflow_model_version: The version of the model.
        input_: The dataframe with the input data.
    """
    filtered_input = {}
    for feature in spec.dataset.feature_columns:
        filtered_input[feature.name] = input_[feature.name]
    input_ = filtered_input

    client = mlflow.MlflowClient()
    model_uri = f"models:/{mlflow_model_name}/{mlflow_model_version}"
    modelversion = client.get_model_version(
        mlflow_model_name, mlflow_model_version
    )
    assert modelversion.run_id, (
        f"missing run_id model with version {mlflow_model_version} at model",
        f"{mlflow_model_name} at Mlflow",
    )
    pipeline = load_pipeline(modelversion.run_id)

    if isinstance(spec, SklearnModelSpec):
        if isinstance(spec.dataset, DatasetConfigWithPreprocessing):
            spec.dataset = spec.dataset.to_dataset_config()
        model = mlflow.sklearn.load_model(model_uri)
        functions = SciKitFunctions(
            spec, None, model=model, preprocessing_pipeline=pipeline
        )
        return functions.predict(input_)

    elif isinstance(spec, TorchModelSpec):
        model = mlflow.pytorch.load_model(model_uri)
        if not isinstance(input_, pd.DataFrame):
            input_ = pd.DataFrame.from_dict(input_, dtype=float)

        check_input(input_, spec.dataset)

        functions = TorchFunctions(
            spec=spec, model=model, preprocessing_pipeline=pipeline
        )
        return functions.predict(input_)
