from datetime import datetime

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
    

