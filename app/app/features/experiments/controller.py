import logging
from asyncio.tasks import Task
from typing import Any, Dict, List, Literal, Optional

import mlflow
from mlflow.tracking._tracking_service.utils import get_tracking_uri
from mlflow.tracking.artifact_utils import get_artifact_uri
from mlflow.tracking.client import MlflowClient
from sqlalchemy.orm.session import Session

from app.api.websocket import WebSocketMessage, get_websockets_manager
from app.builder.dataset import DataModule
from app.features.dataset.crud import repo as dataset_repo
from app.features.experiments.crud import (
    ExperimentCreateRepo,
    ExperimentUpdateRepo,
)
from app.features.experiments.crud import repo as experiments_repo
from app.features.experiments.exceptions import ExperimentNotFound
from app.features.experiments.schema import (
    Experiment,
    ListExperimentsQuery,
    RunningHistory,
    TrainingRequest,
)
from app.features.experiments.tasks import ExperimentView, get_exp_manager
from app.features.experiments.train.custom_logger import AppLogger
from app.features.experiments.train.run import start_training
from app.features.model.crud import repo as model_repo
from app.features.model.exceptions import ModelNotFound, ModelVersionNotFound
from app.features.model.schema.model import ModelVersion
from app.features.user.model import User as UserEntity
from app.schemas.api import ApiBaseModel


def log_error(msg: str):
    logging.error("[experiments_ctl]: %s " % msg)


async def create_model_traning(
    db: Session, user: UserEntity, training_request: TrainingRequest
) -> Experiment:
    model_version = model_repo.get_model_version(
        db, id=training_request.model_version_id
    )

    if not model_version:
        raise ModelVersionNotFound()
    model_version = ModelVersion.from_orm(model_version)

    dataset = dataset_repo.get_by_name(db, model_version.config.dataset.name)
    torchmodel = model_version.build_torch_model()
    featurizers_config = model_version.config.featurizers
    df = dataset.get_dataframe()
    data_module = DataModule(
        featurizers_config=featurizers_config,
        data=df,
        dataset_config=model_version.config.dataset,
        split_target=dataset.split_target,
        split_type=dataset.split_type,
    )
    mlflow_experiment_id = mlflow.create_experiment(training_request.name)
    experiment = experiments_repo.create(
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
    logger = AppLogger(
        experiment.id, experiment_name=training_request.name, user_id=user.id
    )
    task = await start_training(
        torchmodel, training_request, data_module, loggers=logger
    )

    def finish_task(task: Task, experiment_id: int):
        experiment = experiments_repo.get(db, experiment_id)
        assert experiment
        exception = task.exception()
        done = task.done()
        if done and not exception:
            client = MlflowClient()
            model = task.result()
            run = client.create_run(experiment.mlflow_id)
            run_id = run.info.run_id
            mlflow.pytorch.log_model(
                model,
                get_artifact_uri(run_id, tracking_uri=get_tracking_uri()),
                registered_model_name=model_version.mlflow_model_name,
            )
            experiments_repo.update(
                db, obj_in=ExperimentUpdateRepo(stage="SUCCESS"), db_obj=experiment
            )

        elif done and exception:
            experiments_repo.update(
                db, obj_in=ExperimentUpdateRepo(stage="FAILED"), db_obj=experiment
            )
            log_error(str(exception))
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
    model = model_repo.get(db, query.model_id)
    if model and model.created_by_id != user.id:
        raise ModelNotFound()
    exps = experiments_repo.get_by_model_id(db, query.model_id)
    return [Experiment.from_orm(exp) for exp in exps]


def log_metrics(
    db: Session,
    experiment_id: int,
    metrics: dict[str, float],
    history: dict[str, list[float]] = {},
    stage: Literal["train", "val", "test"] = "train",
) -> None:
    print("Hi from log metrics")
    experiment_db = experiments_repo.get(db, experiment_id)
    if not experiment_db:
        raise ExperimentNotFound()

    update_obj = ExperimentUpdateRepo(history=history)
    if stage == "train":
        update_obj.train_metrics = metrics

    experiments_repo.update(db, db_obj=experiment_db, obj_in=update_obj)


def log_hyperparams(db: Session, experiment_id: int, hyperparams: dict[str, Any]):
    experiment_db = experiments_repo.get(db, experiment_id)
    if not experiment_db:
        raise ExperimentNotFound()
    update_obj = ExperimentUpdateRepo(hyperparams=hyperparams)
    experiments_repo.update(db, db_obj=experiment_db, obj_in=update_obj)


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
