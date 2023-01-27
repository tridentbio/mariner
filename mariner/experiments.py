import asyncio
import logging
from asyncio.tasks import Task
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

import mlflow
import mlflow.entities.model_registry.model_version
import ray
from sqlalchemy.orm.session import Session

from api.websocket import WebSocketMessage, get_websockets_manager
from mariner.core.config import settings
from mariner.db.session import SessionLocal
from mariner.entities.user import User as UserEntity
from mariner.events import EventCreate  # BAD DEPENDENCY
from mariner.exceptions import (
    ExperimentNotFound,
    ModelNotFound,
    ModelVersionNotFound,
)
from mariner.ray_actors.training_actors import TrainingActor
from mariner.schemas.api import ApiBaseModel
from mariner.schemas.dataset_schemas import Dataset
from mariner.schemas.experiment_schemas import (
    Experiment,
    ListExperimentsQuery,
    RunningHistory,
    TrainingRequest,
)
from mariner.schemas.model_schemas import ModelVersion, ModelVersionUpdateRepo
from mariner.stores.dataset_sql import dataset_store
from mariner.stores.experiment_sql import (
    ExperimentCreateRepo,
    ExperimentUpdateRepo,
    experiment_store,
)
from mariner.stores.model_sql import model_store
from mariner.tasks import ExperimentView, get_exp_manager
from model_builder.optimizers import AdamParamsSchema, SGDParamsSchema

LOG = logging.getLogger(__name__)


async def make_coroutine_from_ray_objectref(ref: ray.ObjectRef):
    result = await ref
    return result


def handle_training_complete(task: Task, experiment_id: int):
    with SessionLocal() as db:
        experiment = experiment_store.get(db, experiment_id)
        assert experiment
        exception = task.exception()
        done = task.done()
        if not done:
            raise Exception("Task is not done")
        if not exception:
            try:
                best_model_info: mlflow.entities.model_registry.model_version.ModelVersion = (
                    task.result()
                )
                model_store.update_model_version(
                    db,
                    version_id=experiment.model_version_id,
                    obj_in=ModelVersionUpdateRepo(
                        mlflow_version=best_model_info.version
                    ),
                )
            except Exception as exception:
                LOG.error(exception)
                stack_trace = str(exception)
                experiment_store.update(
                    db,
                    obj_in=ExperimentUpdateRepo(stage="ERROR", stack_trace=stack_trace),
                    db_obj=experiment,
                )
            finally:
                experiment_store.update(
                    db, obj_in=ExperimentUpdateRepo(stage="SUCCESS"), db_obj=experiment
                )
                asyncio.ensure_future(
                    get_websockets_manager().send_message(  # noqa
                        user_id=experiment.created_by_id,
                        message=WebSocketMessage(
                            type="update-running-metrics",
                            data=UpdateRunningData(
                                experiment_id=experiment_id,
                                experiment_name=experiment.experiment_name,
                                stage="SUCCESS",
                                running_history=get_exp_manager().get_running_history(
                                    experiment.id
                                ),
                            ),
                        ),
                    )
                )

        else:
            LOG.error(exception)
            stack_trace = str(exception)
            experiment_store.update(
                db,
                obj_in=ExperimentUpdateRepo(stage="ERROR", stack_trace=stack_trace),
                db_obj=experiment,
            )
            asyncio.ensure_future(
                get_websockets_manager().send_message(  # noqa
                    user_id=experiment.created_by_id,
                    message=WebSocketMessage(
                        type="update-running-metrics",
                        data=UpdateRunningData(
                            experiment_id=experiment_id,
                            experiment_name=experiment.experiment_name,
                            stage="ERROR",
                            running_history=get_exp_manager().get_running_history(
                                experiment.id
                            ),
                        ),
                    ),
                )
            )


async def create_model_traning(
    db: Session, user: UserEntity, training_request: TrainingRequest
) -> Experiment:
    """Creates an experiment associated with a model training

    Creates a model training asynchronously through the training actor

    Args:
        db: Session
        user: user that originated request
        training_request: training hyperparams

    Returns:
        experiment created

    Raises:
        ModelVersionNotFound: when model version in training_request is missing
    """
    model_version = model_store.get_model_version(
        db, id=training_request.model_version_id
    )

    if not model_version:
        raise ModelVersionNotFound()

    model_version_parsed = ModelVersion.from_orm(model_version)
    dataset = dataset_store.get_by_name(db, model_version_parsed.config.dataset.name)

    mlflow_experiment_name = f"{training_request.name}-{str(uuid4())}"
    mlflow_experiment_id = mlflow.create_experiment(mlflow_experiment_name)
    experiment = experiment_store.create(
        db,
        obj_in=ExperimentCreateRepo(
            mlflow_id=mlflow_experiment_id,
            experiment_name=training_request.name,
            created_by_id=user.id,
            model_version_id=training_request.model_version_id,
            epochs=training_request.epochs,
            stage="RUNNING",
        ),
    )

    training_actor = TrainingActor.remote(  # type: ignore
        experiment=Experiment.from_orm(experiment),
        request=training_request,
        user_id=user.id,
        mlflow_experiment_name=mlflow_experiment_name,
    )

    # ExperimentManager expect tasks
    training_ref = training_actor.train.remote(dataset=Dataset.from_orm(dataset))
    task = asyncio.create_task(make_coroutine_from_ray_objectref(training_ref))

    get_exp_manager().add_experiment(
        ExperimentView(experiment_id=experiment.id, user_id=user.id, task=task),
        handle_training_complete,
    )
    return Experiment.from_orm(experiment)


def get_experiments(
    db: Session, user: UserEntity, query: ListExperimentsQuery
) -> tuple[List[Experiment], int]:
    """Get's the experiments from ``user``

    Args:
        db: Session
        user: user associated to the request
        query: configures sort, filters and paginations applied

    Returns:
        Tuple where the first element is a list of experiments
        resulted from the pagination and the second is the total
        experiments found in that query (ignoring pagination)

    Raises:
        ModelNotFound: when the query contains a model version not listed
        or not owned by that user

    """
    model = model_store.get(db, query.model_id)
    if model and model.created_by_id != user.id:
        raise ModelNotFound()
    exps, total = experiment_store.get_experiments_paginated(
        db,
        created_by_id=user.id,
        model_id=query.model_id,
        page=query.page,
        per_page=query.per_page,
        stages=query.stage or [],
        order_by=query.order_by,
    )
    return Experiment.from_orm_array(exps), total


def log_metrics(
    db: Session,
    experiment_id: int,
    metrics: dict[str, float],
    history: dict[str, list[float]] = {},
    stage: Literal["train", "val", "test"] = "train",
) -> None:
    experiment_db = experiment_store.get(db, experiment_id)
    if not experiment_db:
        raise ExperimentNotFound()

    update_obj = ExperimentUpdateRepo(history=history)
    if stage == "train":
        update_obj.train_metrics = metrics

    experiment_store.update(db, db_obj=experiment_db, obj_in=update_obj)
    import mariner.events as events_ctl

    model = experiment_db.model_version.model
    events_ctl.create_event(
        db,
        EventCreate(
            source="training:completed",
            user_id=experiment_db.created_by_id,
            timestamp=datetime.now(),
            payload={
                "id": experiment_db.id,
                "experiment_name": experiment_db.experiment_name,
            },
            url=f"{settings.WEBAPP_URL}/models/{model.id}#training",
        ),
    )


def log_hyperparams(db: Session, experiment_id: int, hyperparams: dict[str, Any]):
    experiment_db = experiment_store.get(db, experiment_id)
    if not experiment_db:
        raise ExperimentNotFound()
    update_obj = ExperimentUpdateRepo(hyperparams=hyperparams)
    experiment_store.update(db, db_obj=experiment_db, obj_in=update_obj)


def get_running_histories(
    user: UserEntity, experiment_id: Optional[int] = None
) -> List[RunningHistory]:
    experiments = get_exp_manager().get_from_user(user.id)
    return [
        RunningHistory(
            experiment_id=exp.experiment_id,
            user_id=user.id,
            running_history=exp.running_history,
        )
        for exp in experiments
        if experiment_id is not None or exp.experiment_id == experiment_id
    ]


class UpdateRunningData(ApiBaseModel):
    metrics: Optional[Dict[str, float]] = None
    epoch: Optional[int] = None
    experiment_id: int
    experiment_name: str
    stage: Optional[str] = None
    running_history: Optional[Dict[str, List[float]]] = None


async def send_ws_epoch_update(
    user_id: int,
    experiment_id: int,
    experiment_name: str,
    metrics: dict[str, float],
    epoch: Optional[int] = None,
    stage: Optional[str] = None,
):
    running_history = get_exp_manager().get_running_history(experiment_id)
    if running_history is None:
        return
    for metric_name, metric_value in metrics.items():
        if metric_name not in running_history:
            running_history[metric_name] = []
        running_history[metric_name].append(metric_value)
    await get_websockets_manager().send_message(
        user_id,
        WebSocketMessage(
            type="update-running-metrics",
            data=UpdateRunningData(
                experiment_id=experiment_id,
                experiment_name=experiment_name,
                metrics=metrics,
                epoch=epoch,
                stage=stage,
            ),
        ),
    )


class MonitorableMetric(ApiBaseModel):
    key: str
    label: str
    type: Literal["regressor", "classification"]


def get_metrics_for_monitoring() -> List[MonitorableMetric]:
    """Get's options available for model checkpoint metric monitoring

    Returns:
        List[MonitorableMetric]: Description of the metric
    """
    return [
        MonitorableMetric(
            key="val_mse", label="(MSE) Mean Squared Error", type="regressor"
        ),
        MonitorableMetric(
            key="val_mae", label="(MAE) Mean Absolute Error", type="regressor"
        ),
        MonitorableMetric(key="val_ev", label="EV", type="regressor"),
        MonitorableMetric(key="val_mape", label="MAPE", type="regressor"),
        MonitorableMetric(key="val_R2", label="R2", type="regressor"),
        MonitorableMetric(key="val_pearson", label="Pearson", type="regressor"),
        MonitorableMetric(key="val_accuracy", label="Accuracy", type="classification"),
        MonitorableMetric(
            key="val_precision", label="Precision", type="classification"
        ),
        MonitorableMetric(key="val_recall", label="Recall", type="classification"),
        MonitorableMetric(key="val_f1", label="F1", type="classification"),
        MonitorableMetric(
            key="val_confusion_matrix", label="Confusion MAtrix", type="classification"
        ),
    ]


def get_optimizer_options():
    """Gets optimizer configurations

    Returns what params are needed for each type of optimizer supported
    """
    return [
        AdamParamsSchema(),
        SGDParamsSchema(),
    ]
