from datetime import datetime
from typing import Dict, List

from app.features.model.schema.model import ModelVersion
from app.schemas.api import ApiBaseModel


class TrainingRequest(ApiBaseModel):
    model_name: str
    experiment_name: str
    model_version: str
    learning_rate: float
    epochs: int


class Experiment(ApiBaseModel):
    model_name: str
    model_version_name: str
    model_version: ModelVersion
    created_at: datetime
    updated_at: datetime
    experiment_id: str


class ListExperimentsQuery(ApiBaseModel):
    model_name: str


class RunningHistory(ApiBaseModel):
    experiment_id: str
    user_id: int
    # maps metric name to values
    running_history: Dict[str, List[float]]


class EpochUpdate(ApiBaseModel):
    experiment_id: str
    metrics: Dict[str, float]
