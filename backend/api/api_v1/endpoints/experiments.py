"""
Handlers for api/v1/experiments* endpoints
"""
from typing import Any, List

from fastapi.param_functions import Depends
from fastapi.routing import APIRouter
from sqlalchemy.orm.session import Session

import mariner.experiments as experiments_ctl
from api import deps
from mariner.entities.user import User
from mariner.schemas.api import ApiBaseModel, Paginated
from mariner.schemas.experiment_schemas import (
    Experiment,
    ListExperimentsQuery,
    RunningHistory,
    TrainingRequest,
)
from fleet.model_builder.optimizers import OptimizerSchema

router = APIRouter()


@router.post("/", response_model=Experiment)
async def post_experiments(
    request: TrainingRequest,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_user),
) -> Experiment:
    """Endpoint to create a new training experiment.

    Args:
        request: The payload describing the training experiment.
        db: Connection to the database.
        current_user: User that originated the request.

    Returns:
        Created experiment.
    """
    result = await experiments_ctl.create_model_traning(db, current_user, request)
    return result


@router.get("/", response_model=Paginated[Experiment])
def get_experiments(
    experiments_query: ListExperimentsQuery = Depends(),
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_user),
) -> Paginated[Experiment]:
    """Gets a user's experiments.

    Args:
        experiments_query: Query to specify which experiments to search.
        db: Connection to the database.
        current_user: User that originated the request.

    Returns:
        A paginated view of the experiments queried.
    """
    data, total = experiments_ctl.get_experiments(db, current_user, experiments_query)
    return Paginated(data=data, total=total)


@router.get("/running-history", response_model=List[RunningHistory])
def get_experiments_running_history(
    user: User = Depends(deps.get_current_active_user),
) -> List[RunningHistory]:
    """Gets an experiment history of metrics.

    Args:
        user: User that originated request.

    Returns:
        List with the history of metrics for all experiments from user.
    """
    histories = experiments_ctl.get_running_histories(user)
    return histories


class MetricsUpdate(ApiBaseModel):
    """Models the payload to update a metric from a running training."""

    type: str
    data: Any
    experiment_id: int
    experiment_name: str
    user_id: int


@router.post(
    "/epoch_metrics",
    response_model=str,
    dependencies=[Depends(deps.assert_trusted_service)],
)
async def post_update_metrics(
    parsed_msg: MetricsUpdate, db: Session = Depends(deps.get_db)
):
    """Updates experiment with metrics of a step.

    Function used by custom logger to remotely log metrics.

    Args:
        parsed_msg: The payload with metrics.
        db: Connection with database.

    Raises:
        ValueError: when the type of parsed_msg is unhandled.
    """
    msgtype = parsed_msg.type
    data = parsed_msg.data
    experiment_id = parsed_msg.experiment_id
    experiment_name = parsed_msg.experiment_name
    user_id = parsed_msg.user_id
    if msgtype == "epochMetrics":
        await experiments_ctl.send_ws_epoch_update(
            user_id=user_id,
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            metrics=data["metrics"],
            epoch=data["epoch"],
        )
    elif msgtype == "metrics":
        history = data["history"]
        metrics = data["metrics"]
        experiments_ctl.log_metrics(
            db=db, experiment_id=experiment_id, metrics=metrics, history=history
        )
        await experiments_ctl.send_ws_epoch_update(
            experiment_id=experiment_id,
            metrics=metrics,
            experiment_name=experiment_name,
            user_id=user_id,
            stage="SUCCESS",
        )
    elif msgtype == "hyperparams":
        # Don't need to save the config argument of the CustomModel again
        if "config" in data:
            data.pop("config")
        experiments_ctl.log_hyperparams(
            db=db,
            experiment_id=experiment_id,
            hyperparams=data,
        )
    else:
        raise ValueError(f"Failed msg type {msgtype}")
    return "ok"


@router.get(
    "/metrics",
    dependencies=[Depends(deps.get_current_active_user)],
    response_model=List[experiments_ctl.MonitorableMetric],
)
def get_experiments_metrics():
    """Gets monitorable metrics to configure early stopping and checkpoint monitoring"""
    return experiments_ctl.get_metrics_for_monitoring()


@router.get(
    "/optimizers",
    dependencies=[Depends(deps.get_current_active_user)],
    response_model=List[OptimizerSchema],
)
def get_training_experiment_optimizers():
    """Gets the options for experiment optimizers."""
    return experiments_ctl.get_optimizer_options()


@router.get(
    "/{model_version_id}/metrics",
    response_model=List[Experiment],
)
def get_experiments_metrics_for_model_version(
    model_version_id: int,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_user),
):
    """Gets the experiments with metrics for a model version."""
    return experiments_ctl.get_experiments_metrics_for_model_version(
        db=db, model_version_id=model_version_id, user=current_user
    )
