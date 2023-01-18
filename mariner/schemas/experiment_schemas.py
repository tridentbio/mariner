from typing import Dict, List, Literal, Optional, Union

from fastapi import Depends, Query
from pydantic import Required

from mariner.schemas.api import (
    ApiBaseModel,
    OrderByQuery,
    get_order_by_query,
    utc_datetime,
)
from mariner.schemas.model_schemas import ModelVersion
from mariner.schemas.user_schemas import User


class MonitoringConfig(ApiBaseModel):
    """
    Configures model checkpointing
    """

    metric_key: str
    mode: str


class TrainingRequest(ApiBaseModel):
    name: str
    model_version_id: int
    learning_rate: float
    epochs: int
    batch_size: Optional[int] = None
    monitoring_config: MonitoringConfig


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


class ListExperimentsQuery:
    def __init__(
        self,
        stage: Union[list[str], None] = Query(default=None),
        model_id: Union[int, None] = Query(default=Required, alias="modelId"),
        model_version_ids: Union[list[int], None] = Query(
            default=None, alias="modelVersionIds"
        ),
        page: int = Query(default=0),
        per_page: int = Query(default=15, alias="perPage"),
        order_by: Union[OrderByQuery, None] = Depends(get_order_by_query),
    ):
        self.stage = stage
        self.model_id = model_id
        self.page = page
        self.per_page = per_page
        self.model_version_ids = model_version_ids
        self.order_by = order_by


class RunningHistory(ApiBaseModel):
    experiment_id: int
    user_id: int
    # maps metric name to values
    running_history: Dict[str, List[float]]


class EpochUpdate(ApiBaseModel):
    experiment_id: str
    metrics: Dict[str, float]
