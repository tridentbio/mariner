"""
The fleet.models module is responsible to export abstractions
around different machine learning algorithms provided by 
underlying packages such as torch and sci-kit
"""


from dataclasses import dataclass
from typing import Union

import mlflow
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from pandas import DataFrame
from pydantic import BaseModel

from fleet.base_schemas import BaseFleetModelSpec, BaseModelFunctions
from fleet.scikit_.model_functions import SciKitFunctions
from fleet.scikit_.schemas import SciKitModelSpec, SciKitTrainingConfig
from fleet.torch_.model_functions import TorchFunctions
from fleet.torch_.schemas import TorchModelSpec, TorchTrainingConfig


@dataclass
class Result:
    mlflow_experiment_id: str


def fit(
    spec: BaseFleetModelSpec,
    train_config: BaseModel,  # todo: make this type narrower
    dataset: DataFrame,
    mlflow_model_name: str,
    mlflow_experiment_name: str,
    datamodule_args: dict = {},
):
    functions: Union[BaseModelFunctions, None] = None
    mlflow_experiment_id = mlflow.create_experiment(mlflow_experiment_name)
    if isinstance(spec, TorchModelSpec):
        functions = TorchFunctions()
        functions.loggers = [MLFlowLogger(experiment_name=mlflow_experiment_name)]
        assert isinstance(train_config, TorchTrainingConfig)
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

    functions.log_models(
        mlflow_model_name=mlflow_model_name,
        mlflow_experiment_id=mlflow_experiment_id,
    )
    return Result(mlflow_experiment_id=mlflow_experiment_id)
