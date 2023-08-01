"""
Experiments service
"""
import asyncio
import logging
from asyncio.tasks import Task
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

import ray
from sqlalchemy.orm.session import Session

import mariner.events as events_ctl
from api.websocket import WebSocketMessage, get_websockets_manager
from fleet.model_builder.optimizers import (
    AdamParamsSchema,
    OptimizerSchema,
    SGDParamsSchema,
)
from fleet.model_functions import Result
from fleet.ray_actors.training_actors import TrainingActor
from mariner.core.aws import Bucket
from mariner.core.config import get_app_settings
from mariner.db.session import SessionLocal
from mariner.entities.user import User as UserEntity
from mariner.events import EventCreate  # BAD DEPENDENCY
from mariner.exceptions import (
    ExperimentNotFound,
    ModelNotFound,
    ModelVersionNotFound,
)
from mariner.schemas.api import ApiBaseModel
from mariner.schemas.experiment_schemas import (
    BaseTrainingRequest,
    Experiment,
    ListExperimentsQuery,
    RunningHistory,
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

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


async def make_coroutine_from_ray_objectref(ref: ray.ObjectRef):
    """Transforms the ray into a coroutine
    Args:
        ref: ray ObjectRef
    """
    result = await ref
    return result


def handle_training_complete(task: Task, experiment_id: int):
    """Completes the training process making the needed db updates

    Updates model version with resulting best model on validation metric
    and experiment status as finished

    Args:
        task: asyncio.Task wrapping training result
        experiment_id: id of the experiment that is being completed

    Raises:
        RuntimeError: when function is called on a task that is not finished
    """
    with SessionLocal() as db:
        experiment = experiment_store.get(db, experiment_id)
        assert experiment
        exception = task.exception()
        done = task.done()
        if not done:
            raise RuntimeError("Task is not done")
        if exception:
            LOG.error(exception)
            stack_trace = str(exception)
            experiment_store.update(
                db,
                obj_in=ExperimentUpdateRepo(
                    stage="ERROR",
                    stack_trace=stack_trace,
                    updated_at=datetime.now().isoformat(),
                ),
                db_obj=experiment,
            )
            asyncio.ensure_future(
                get_websockets_manager().send_message_to_user(  # noqa
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
            return
        result: Result = task.result()
        model_store.update_model_version(
            db,
            version_id=experiment.model_version_id,
            obj_in=ModelVersionUpdateRepo(
                mlflow_version=result.mlflow_model_version.version
            ),
        )
        experiment_store.update(
            db,
            obj_in=ExperimentUpdateRepo(
                mlflow_id=result.mlflow_experiment_id,
                stage="SUCCESS",
                updated_at=datetime.now().isoformat(),
            ),
            db_obj=experiment,
        )
        asyncio.ensure_future(
            get_websockets_manager().send_message_to_user(
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


async def create_model_training(
    db: Session, user: UserEntity, training_request: BaseTrainingRequest
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

    experiment = experiment_store.create(
        db,
        obj_in=ExperimentCreateRepo(
            experiment_name=training_request.name,
            created_by_id=user.id,
            model_version_id=training_request.model_version_id,
            epochs=training_request.config.epochs,
            hyperparams={"learning_rate": training_request.config.optimizer.params.lr},
            stage="RUNNING",
        ),
    )

    training_actor = TrainingActor.remote(  # type: ignore
        experiment=Experiment.from_orm(experiment),
        request=training_request,
        user_id=user.id,
        mlflow_experiment_name=mlflow_experiment_name,
    )

    dataset_uri = f"s3://{Bucket.Datasets.value}/{dataset.data_url}"
    training_ref = training_actor.fit.remote(
        experiment_id=experiment.id,
        experiment_name=experiment.experiment_name,
        user_id=user.id,
        spec=model_version_parsed.config,
        train_config=training_request.config,
        dataset_uri=dataset_uri,
        mlflow_model_name=model_version.mlflow_model_name,
        mlflow_experiment_name=mlflow_experiment_name,
        datamodule_args={
            "split_target": dataset.split_target,
            "split_type": dataset.split_type,
        },
    )
    get_exp_manager().add_experiment(
        ExperimentView(
            experiment_id=experiment.id,
            user_id=user.id,
            task=asyncio.create_task(make_coroutine_from_ray_objectref(training_ref)),
        ),
        handle_training_complete,
    )
    return Experiment.from_orm(experiment)


def get_experiments(
    db: Session, user: UserEntity, query: ListExperimentsQuery
) -> tuple[List[Experiment], int]:
    """Gets the experiments from ``user``.

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


def get_experiment(db: Session, user: UserEntity, experiment_id: int) -> Experiment:
    """
    Gets the experiment from ``user``.

    Args:
        db: Session
        user: user associated to the request
        experiment_id: id of the experiment

    Returns:
        Experiment

    Raises:
        ExperimentNotFound: when the experiment is not found
        or not owned by that user
    """
    experiment = experiment_store.get(db, experiment_id)
    if not experiment or experiment.created_by_id != user.id:
        raise ExperimentNotFound()
    return Experiment.from_orm(experiment)


def log_metrics(
    db: Session,
    experiment_id: int,
    metrics: dict[str, float],
    history: Union[dict[str, list[float]], None] = None,
    stage: Literal["train", "val", "test"] = "train",
) -> None:
    """Saves the metrics reported of a modelversion being trained to the database

    This function is used by the ``CustomLogger`` to make updates to the modelversion
    metrics

    Args:
        db: connection with the database
        experiment_id: id of the experiment
        metrics: dict with metrics
        history: history of the metrics
        stage

    Raises:
        ExperimentNotFound: if no experiment is found with id ``experiment_id``
    """
    experiment_db = experiment_store.get(db, experiment_id)
    if not experiment_db:
        raise ExperimentNotFound()

    update_obj = ExperimentUpdateRepo(history=history)
    if stage == "train":
        update_obj.train_metrics = metrics

    experiment_store.update(db, db_obj=experiment_db, obj_in=update_obj)

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
            url=f"{get_app_settings('webapp').url}/models/{model.id}#training",
        ),
    )


def log_hyperparams(db: Session, experiment_id: int, hyperparams: dict[str, Any]):
    """Saves the hyperparameters logged by ``CustomLogger``

    Args:
        db: connection with the database
        experiment_id: id of the experiment
        hyperparams: dictionary with hyperparameter values

    Raises:
        ExperimentNotFound: when experiment with id ``experiment_id`` is missing
    """
    experiment_db = experiment_store.get(db, experiment_id)
    if not experiment_db:
        raise ExperimentNotFound()
    update_obj = ExperimentUpdateRepo(hyperparams=hyperparams)
    experiment_store.update(db, db_obj=experiment_db, obj_in=update_obj)


def get_running_histories(
    user: UserEntity, experiment_id: Optional[int] = None
) -> List[RunningHistory]:
    """Gets metrics history from a users experiment that is running

    Gets the metrics from ingoing trainings that are persisted to the
    ``ExperimentManager`` singleton

    Args:
        user
        experiment_id

    Returns:
        List[RunningHistory]
    """
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
    """The metrics gathered in ``epoch`` training iteration"""

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
    """Streams the metrics updates to the user

    Sends the metrics from the model being trained at a specific epoch
    to the user with id ``user_id`` through websocket


    Args:
        user_id: id of the user that created the experiment
        experiment_id: id of the experiment
        experiment_name: name of the experiment
        metrics: dictionary with epoch metrics
        epoch: epoch
        stage: train/val/test
    """
    running_history = get_exp_manager().get_running_history(experiment_id)
    if running_history is None:
        return
    for metric_name, metric_value in metrics.items():
        if metric_name not in running_history:
            running_history[metric_name] = []
        running_history[metric_name].append(metric_value)
    await get_websockets_manager().send_message_to_user(
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
    """
    Metric that can be used as a monitoring argument for training
    callbacks
    """

    key: str
    label: str
    tex_label: Union[str, None] = None
    type: Literal["regressor", "classification"]


def get_metrics_for_monitoring() -> List[MonitorableMetric]:
    """Gets options available for model checkpoint metric monitoring

    The real metric keys have the stage prefixed, e.g.: ``train_mse``, ``val_f1``

    Returns:
        List[MonitorableMetric]: Description of the metric
    """
    return [
        MonitorableMetric(key="mse", label="MSE", type="regressor"),
        MonitorableMetric(key="mae", label="MAE", type="regressor"),
        MonitorableMetric(key="ev", label="EV", type="regressor"),
        MonitorableMetric(key="mape", label="MAPE", type="regressor"),
        MonitorableMetric(key="R2", label="R2", tex_label="R^2", type="regressor"),
        MonitorableMetric(key="pearson", label="Pearson", type="regressor"),
        MonitorableMetric(key="accuracy", label="Accuracy", type="classification"),
        MonitorableMetric(key="precision", label="Precision", type="classification"),
        MonitorableMetric(key="recall", label="Recall", type="classification"),
        MonitorableMetric(key="f1", label="F1", type="classification"),
        # removed on UI fixes - 2/6/2023:
        # MonitorableMetric(
        #     key="confusion_matrix", label="Confusion Matrix", type="classification"
        # ),
    ]


def get_optimizer_options() -> List[OptimizerSchema]:
    """Gets optimizer configurations

    Returns what params are needed for each type of optimizer supported
    """
    return [
        AdamParamsSchema(),
        SGDParamsSchema(),
    ]


def get_experiments_metrics_for_model_version(
    db: Session, model_version_id: int, user: UserEntity
) -> List[Experiment]:
    """Gets the experiments for a model version

    Args:
        db: connection with the database
        model_version_id: id of the model version
        user: user that is requesting the experiments

    Returns:
        List[Experiment]
    """
    return experiment_store.get_experiments_metrics_for_model_version(
        db, model_version_id, user.id
    )
