from datetime import datetime
from typing import Dict, List, Optional

from app.features.model.schema.model import ModelVersion
from app.features.user.schema import User
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
    created_by_id: int
    experiment_id: str
    stage: str
    created_by: Optional[User] = None
    hyperparams: Optional[Dict[str, float]] = None
    epochs: Optional[int] = None
    train_metrics: Optional[Dict[str, float]] = None
    val_metrics: Optional[Dict[str, float]] = None
    test_metrics: Optional[Dict[str, float]] = None
    history: Optional[Dict[str, List[float]]] = None


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
