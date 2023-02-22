"""
Experiment related DTOs
"""
from typing import Dict, List, Literal, Optional, Union

from fastapi import Depends, Query

from mariner.schemas.api import (
    ApiBaseModel,
    OrderByQuery,
    get_order_by_query,
    utc_datetime,
)
from mariner.schemas.model_schemas import ModelVersion
from mariner.schemas.user_schemas import User
from model_builder.optimizers import Optimizer


class MonitoringConfig(ApiBaseModel):
    """
    Configures model checkpointing
    """

    metric_key: str
    mode: str


class EarlyStoppingConfig(ApiBaseModel):
    """
    Configures earlystopping of training
    """

    metric_key: str
    mode: str
    min_delta: float = 5e-2
    patience: int = 5
    check_finite: bool = False


class TrainingRequest(ApiBaseModel):
    """
    Configure options for starting a training
    """

    name: str
    model_version_id: int
    epochs: int
    batch_size: Optional[int] = None
    checkpoint_config: MonitoringConfig
    optimizer: Optimizer
    early_stopping_config: Optional[EarlyStoppingConfig]


ExperimentStage = Literal["NOT RUNNING", "RUNNING", "SUCCESS", "ERROR"]


class Experiment(ApiBaseModel):
    """
    Experiment entry
    """

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
    hyperparams: Optional[Dict[str, Union[float, None]]] = None
    epochs: Optional[int] = None
    train_metrics: Optional[Dict[str, float]] = None
    val_metrics: Optional[Dict[str, float]] = None
    test_metrics: Optional[Dict[str, float]] = None
    history: Optional[Dict[str, List[float]]] = None
    stack_trace: Optional[str]


class ListExperimentsQuery:
    """
    Used to get the listing experiments query from the querystring
    """

    def __init__(
        self,
        stage: Union[list[str], None] = Query(default=None),
        model_id: Union[int, None] = Query(default=None, alias="modelId"),
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
    """
    Objects used to update the frontend progress bar of a training.
    Sends complete metrics history for a client to catch up (in case
    of missing EpochUpdates)
    """

    experiment_id: int
    user_id: int
    # maps metric name to values
    running_history: Dict[str, List[float]]


class EpochUpdate(ApiBaseModel):
    """
    Update the client with the metrics gotten from a single training step
    """

    experiment_id: str
    metrics: Dict[str, float]
