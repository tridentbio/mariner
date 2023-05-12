"""
The fleet.models module is responsible to export abstractions
around different machine learning algorithms provided by 
underlying packages such as torch and sci-kit
"""


import logging
from dataclasses import dataclass
from typing import List, Union

import mlflow
import pandas as pd
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from mlflow.entities.model_registry.model_version import ModelVersion
from pandas import DataFrame
from pydantic import BaseModel

from fleet.base_schemas import (
    BaseFleetModelSpec,
    BaseModelFunctions,
    TorchModelSpec,
)
from fleet.scikit_.model_functions import SciKitFunctions
from fleet.scikit_.schemas import SciKitModelSpec, SciKitTrainingConfig
from fleet.torch_.model_functions import TorchFunctions
from fleet.torch_.schemas import TorchTrainingConfig
from mariner.core.aws import Bucket, download_file_as_dataframe
from mariner.train.custom_logger import MarinerLogger

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


@dataclass
class Result:
    mlflow_experiment_id: str
    mlflow_model_version: ModelVersion


def fit(
    spec: BaseFleetModelSpec,
    train_config: BaseModel,  # todo: make this type narrower
    mlflow_model_name: str,
    mlflow_experiment_name: str,
    experiment_id: Union[None, int] = None,
    experiment_name: Union[None, str] = None,
    user_id: Union[None, int] = None,
    datamodule_args: dict = {},
    dataset_uri: Union[None, str] = None,
    dataset: Union[None, DataFrame] = None,
):
    functions: Union[BaseModelFunctions, None] = None
    try:
        mlflow_experiment_id = mlflow.create_experiment(mlflow_experiment_name)
    except Exception as exp:
        LOG.error("%r", exp)
        raise RuntimeError("Failed to create mlflow experiment")

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
            datamodule_args=datamodule_args,
        )
    elif isinstance(spec, SciKitModelSpec):
        functions = SciKitFunctions()
        assert isinstance(train_config, SciKitTrainingConfig)

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
