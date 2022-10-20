from typing import Dict, List, Literal, Optional

from fastapi import Query

from mariner.schemas.model_schemas import ModelVersion
from mariner.schemas.user_schemas import User
from mariner.schemas.api import ApiBaseModel, utc_datetime


class TrainingRequest(ApiBaseModel):
    name: str
    model_version_id: int
    learning_rate: float
    epochs: int
    batch_size: Optional[int] = None


ExperimentStage = Literal["NOT RUNNING", "RUNNING", "SUCCESS", "ERROR"]


class Experiment(ApiBaseModel):
    experiment_name: Optional[str]
    model_version_id: int
    model_version: ModelVersion
    created_at: utc_datetime
    updated_at: utc_datetime
    created_by_id: int
    id: int
    mlflow_id: str
    stage: ExperimentStage
    created_by: Optional[User] = None
    hyperparams: Optional[Dict[str, float]] = None
    epochs: Optional[int] = None
    train_metrics: Optional[Dict[str, float]] = None
    val_metrics: Optional[Dict[str, float]] = None
    test_metrics: Optional[Dict[str, float]] = None
    history: Optional[Dict[str, List[float]]] = None
    stack_trace: Optional[str]


class ListExperimentsQuery(ApiBaseModel):
    stage: Optional[List[str]] = Query(default=None)
    model_id: int
    model_version_ids: Optional[List[int]] = Query(default=None)


class RunningHistory(ApiBaseModel):
    experiment_id: int
    user_id: int
    # maps metric name to values
    running_history: Dict[str, List[float]]


class EpochUpdate(ApiBaseModel):
    experiment_id: str
    metrics: Dict[str, float]
