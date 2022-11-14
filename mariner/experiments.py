import logging
from asyncio.tasks import Task
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

import mlflow
from pytorch_lightning.loggers import MLFlowLogger
from mlflow.tracking._tracking_service.utils import get_tracking_uri
from mlflow.tracking.artifact_utils import get_artifact_uri
from mlflow.tracking.client import MlflowClient
from sqlalchemy.orm.session import Session

from api.websocket import WebSocketMessage, get_websockets_manager
from mariner.core.config import settings
from mariner.entities.user import User as UserEntity
from mariner.events import EventCreate  # BAD DEPENDENCY
from mariner.exceptions import (
    ExperimentNotFound,
    ModelNotFound,
    ModelVersionNotFound,
)
from mariner.schemas.api import ApiBaseModel
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
from mariner.train.custom_logger import AppLogger
from mariner.train.run import start_training
from model_builder.dataset import DataModule


def log_error(msg: str):
    logging.error("[experiments_ctl]: %s " % msg)


async def create_model_traning(
    db: Session, user: UserEntity, training_request: TrainingRequest
) -> Experiment:
    model_version = model_store.get_model_version(
        db, id=training_request.model_version_id
    )

    if not model_version:
        raise ModelVersionNotFound()
    model_version = ModelVersion.from_orm(model_version)

    dataset = dataset_store.get_by_name(db, model_version.config.dataset.name)
    torchmodel = model_version.build_torch_model()
    featurizers_config = model_version.config.featurizers
    df = dataset.get_dataframe()
    data_module = DataModule(
        featurizers_config=featurizers_config,
        data=df,
        dataset_config=model_version.config.dataset,
        split_target=dataset.split_target,
        split_type=dataset.split_type,
        batch_size=training_request.batch_size or 32,
    )
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
    logger = [
        AppLogger(
            experiment.id, experiment_name=training_request.name, user_id=user.id
        ),
        MLFlowLogger(experiment_name=experiment.experiment_name),
    ]
    task = await start_training(
        torchmodel, training_request, data_module, loggers=logger
    )

    def finish_task(task: Task, experiment_id: int):
        experiment = experiment_store.get(db, experiment_id)
        assert experiment
        exception = task.exception()
        done = task.done()
        if done and not exception:
            client = MlflowClient()
            model = task.result()
            runs = client.search_runs(experiment_ids=[experiment.mlflow_id])
            assert len(runs) == 1
            run = runs[0]
            run_id = run.info.run_id
            model_info = mlflow.pytorch.log_model(
                model,
                get_artifact_uri(run_id, tracking_uri=get_tracking_uri()),
                registered_model_name=model_version.mlflow_model_name,
            )
            mlflow_model = client.get_registered_model(model_version.mlflow_model_name)
            assert (
                mlflow_model.latest_versions and len(mlflow_model.latest_versions) > 0
            ), "runtime error: latest_versions should have at least a version"
            latest_version = mlflow_model.latest_versions[-1]
            assert latest_version
            model_info.mlflow_version
            experiment_store.update(
                db, obj_in=ExperimentUpdateRepo(stage="SUCCESS"), db_obj=experiment
            )
            model_store.update_model_version(
                db,
                experiment.model_version_id,
                ModelVersionUpdateRepo(mlflow_version=latest_version.version),
            )

        elif done and exception:
            stack_trace = str(exception)
            experiment_store.update(
                db,
                obj_in=ExperimentUpdateRepo(stage="FAILED", stack_trace=stack_trace),
                db_obj=experiment,
            )
            logging.error(exception)
        else:
            raise Exception("Task is not done")

    get_exp_manager().add_experiment(
        ExperimentView(experiment_id=experiment.id, user_id=user.id, task=task),
        finish_task,
    )
    return Experiment.from_orm(experiment)


def get_experiments(
    db: Session, user: UserEntity, query: ListExperimentsQuery
) -> List[Experiment]:
    model = model_store.get(db, query.model_id)
    if model and model.created_by_id != user.id:
        raise ModelNotFound()
    exps = experiment_store.get_many(db, query)
    return [Experiment.from_orm(exp) for exp in exps]


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


def get_running_histories(user: UserEntity) -> List[RunningHistory]:
    experiments = get_exp_manager().get_from_user(user.id)
    return [
        RunningHistory(
            experiment_id=exp.experiment_id,
            user_id=user.id,
            running_history=exp.running_history,
        )
        for exp in experiments
    ]


class UpdateRunningData(ApiBaseModel):
    metrics: Dict[str, float]
    epoch: Optional[int] = None
    experiment_id: int
    experiment_name: str
    stage: Optional[str] = None


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
