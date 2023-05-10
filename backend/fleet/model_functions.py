"""
The fleet.models module is responsible to export abstractions
around different machine learning algorithms provided by 
underlying packages such as torch and sci-kit
"""


from dataclasses import dataclass
from typing import List, Union
from lightning.pytorch.loggers.logger import Logger

import mlflow
from mlflow.entities.model_registry.model_version import ModelVersion
import pandas as pd
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from pandas import DataFrame
from pydantic import BaseModel

from fleet.base_schemas import BaseFleetModelSpec, BaseModelFunctions
from fleet.scikit_.model_functions import SciKitFunctions
from fleet.scikit_.schemas import SciKitModelSpec, SciKitTrainingConfig
from fleet.torch_.model_functions import TorchFunctions
from fleet.torch_.schemas import TorchModelSpec, TorchTrainingConfig
from mariner.train.custom_logger import MarinerLogger

import logging

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
    LOG.error("STARTING FIT")
    functions: Union[BaseModelFunctions, None] = None
    try:
        mlflow_experiment_id = mlflow.create_experiment(mlflow_experiment_name)
    except Exception as exp:
        LOG.error("%r", exp)
        raise RuntimeError("Failed to create mlflow experiment")

    LOG.error("mlflow_experinment_id %r", mlflow_experiment_id)

    if dataset is None and dataset_uri is None:
        LOG.error("MISSING DATASET")
        raise ValueError("dataset_uri or dataset must be passed to fit()")
    elif dataset is None and dataset_uri is not None:
        LOG.error("READING CSV")
        dataset = pd.read_csv(dataset_uri)
    assert isinstance(dataset, DataFrame), "Dataset must be DataFrame"
    LOG.error("DataFrame loaded")

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

        LOG.error("DataFrame loaded")
        functions.loggers = loggers
        assert isinstance(train_config, TorchTrainingConfig)
        LOG.error("TRAINING NOW")
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

    LOG.error("FINISHED TRAINING!!!!")
    mlflow_model_version = functions.log_models(
        mlflow_model_name=mlflow_model_name,
        mlflow_experiment_id=mlflow_experiment_id,
    )

    LOG.error("FINISHED LOGGING!!!!")

    return Result(
        mlflow_experiment_id=mlflow_experiment_id,
        mlflow_model_version=mlflow_model_version,
    )
