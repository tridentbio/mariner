"""
Models service
"""
import asyncio
import logging
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import mlflow
import mlflow.exceptions
import ray
from sqlalchemy.orm.session import Session

from api.websocket import WebSocketResponse, get_websockets_manager
from fleet import options
from fleet.model_functions import predict
from fleet.ray_actors.model_check_actor import ModelCheckActor
from fleet.ray_actors.tasks import TaskControl, get_task_control
from mariner.core import mlflowapi
from mariner.db.session import SessionLocal
from mariner.entities.user import User as UserEntity
from mariner.exceptions import (
    DatasetNotFound,
    ModelNameAlreadyUsed,
    ModelNotFound,
    ModelVersionNotFound,
)
from mariner.exceptions.model_exceptions import (
    ModelVersionNotTrained,
    ModelVersionNotUpdatable,
)
from mariner.schemas.api import ApiBaseModel
from mariner.schemas.dataset_schemas import Dataset as DatasetSchema
from mariner.schemas.model_schemas import (
    Model,
    ModelCreate,
    ModelCreateRepo,
    ModelFeaturesAndTarget,
    ModelsQuery,
    ModelVersion,
    ModelVersionCreateRepo,
    ModelVersionUpdate,
    ModelVersionUpdateRepo,
)
from mariner.stores.dataset_sql import dataset_store
from mariner.stores.model_sql import model_store

LOG = logging.getLogger(__file__)
LOG.setLevel(logging.INFO)


def _check_model(
    model_version: ModelVersion, user: UserEntity, task_control: TaskControl
):
    ray_task = start_check_model_step_exception(
        model_version=model_version,
        user=user,
        task_control=task_control,
    )
    coroutine = _make_coroutine(ray_task)
    task = asyncio.create_task(coroutine)
    task.add_done_callback(
        lambda _: handle_model_check_finished(
            task_args={"model_version_id": model_version.id},
            task_control=task_control,
        )
    )
    return task


def start_check_model_step_exception(
    model_version: ModelVersion, user: UserEntity, task_control: TaskControl
) -> ray.ObjectRef:
    """Checks the steps of a pytorch lightning model built from config.

    The checking task runs asynchronously in a ray actor. It is added to
    the task control so that it can be monitored.

    Args:
        db: Connection to the database, needed to get the dataset.
        config: ModelSchema instance specifying the model.

    Returns:
        An object either with the stack trace or the model output.

    See also:
        :class:`fleet.ray_actors.model_check_actor.ModelCheckActor`
    """
    with SessionLocal() as db:
        dataset = DatasetSchema.from_orm(
            dataset_store.get_by_name(
                db,
                model_version.config.dataset.name,
                user_id=user.id,
            )
        )
        actor = ModelCheckActor.remote()  # pylint: disable=E1101
        task = actor.check_model_steps.remote(  # pylint: disable=E1101
            dataset=dataset, config=model_version.config
        )
        task_control.add_task(
            task,
            {
                "model_version_id": model_version.id,
            },
        )

        return task


def handle_model_check_finished(
    task_args: Dict[str, Any],
    task_control: TaskControl,
):
    """Handles the result of the model check task.

    1. Update model version check_status
    fields according to task result.
    2. Remove task from task control.
    3. Update connected users of interest
    """
    # Update model version check status
    model_version_id = task_args["model_version_id"]
    ids, tasks = task_control.get_tasks(
        {
            "model_version_id": model_version_id,
        }
    )
    if len(tasks) == 0:
        LOG.warning("Handling check but no tasks were found")
    with SessionLocal() as db:
        for id_, task in zip(ids, tasks):
            result = ray.get(task)
            model_version = model_store.update_model_version(
                db=db,
                version_id=task_args["model_version_id"],
                obj_in=ModelVersionUpdateRepo(
                    check_status="OK" if not result.stack_trace else "FAILED",
                    check_stack_trace=result.stack_trace,
                ).dict(exclude_unset=True),
            )
            if not model_version:
                raise RuntimeError(
                    "model_version is None: Failed to update model version"
                )

            # User of interest is the model version creator
            users = [model_version.created_by_id]
            asyncio.ensure_future(
                get_websockets_manager().send_message_to_user(
                    user_id=users,
                    message=WebSocketResponse(
                        type="update-model",
                        data=ModelVersion.from_orm(model_version),
                    ),
                )
            )

        # Remove tasks:
        for id_ in ids:
            task_control.remove_task(id_)


async def _make_coroutine(ray_object_ref: ray.ObjectRef):
    """
    Turns a ray task into a coroutine.
    """
    result = await ray_object_ref
    return result


async def create_model(
    db: Session,
    user: UserEntity,
    model_create: ModelCreate,
    return_async_task=False,
) -> Model | Tuple[Model, Any]:
    r"""
    Creates a model and a model version if passed with a
    non-existing `model_create.name`, otherwise creates a
    new model version.
    Triggers the creation of a RegisteredModel and a ModelVersion
    in MLFlow API's
    """
    dataset = dataset_store.get_by_name(
        db, model_create.config.dataset.name, user_id=user.id
    )
    if not dataset:
        raise DatasetNotFound(
            f"Dataset {model_create.config.dataset.name} not found"
        )

    client = mlflowapi.create_tracking_client()
    # Handle case where model_create.name refers to existing model
    existingmodel = model_store.get_by_name_from_user(
        db, model_create.name, user_id=user.id
    )
    if existingmodel:
        check_status = (
            "OK" if model_create.config.framework != "torch" else "RUNNING"
        )
        model_version = model_store.create_model_version(
            db,
            ModelVersionCreateRepo(
                mlflow_version=None,
                mlflow_model_name=existingmodel.mlflow_name,
                model_id=existingmodel.id,
                name=model_create.config.name,
                config=model_create.config,
                created_by_id=user.id,
                description=model_create.model_version_description,
                check_status=check_status,
            ),
        )
        task = None
        if check_status == "RUNNING":
            task = _check_model(
                model_version=ModelVersion.from_orm(model_version),
                user=user,
                task_control=get_task_control(),
            )
        model = Model.from_orm(existingmodel)
        if return_async_task:
            return model, task
        return model

    # Handle cases of creating a new model in the registry
    mlflow_name = f"{user.id}-{model_create.name}-{uuid4()}"

    try:
        mlflowapi.create_registered_model(
            client,
            name=mlflow_name,
            description=model_create.model_description,
        )
    except mlflow.exceptions.MlflowException as exp:
        if exp.error_code == mlflow.exceptions.RESOURCE_ALREADY_EXISTS:
            raise ModelNameAlreadyUsed(
                "A model with that name is already in use by mlflow"
            ) from exp
        raise

    model = model_store.create(
        db,
        obj_in=ModelCreateRepo(
            name=model_create.name,
            description=model_create.model_description,
            mlflow_name=mlflow_name,
            created_by_id=user.id,
            dataset_id=dataset.id,
            columns=[
                ModelFeaturesAndTarget.construct(
                    column_name=feature_col.name,
                    column_type="feature",
                )
                for feature_col in model_create.config.dataset.feature_columns
            ]
            + [
                ModelFeaturesAndTarget.construct(
                    column_name=target_column.name,
                    column_type="target",
                )
                for target_column in model_create.config.dataset.target_columns
            ],
        ),
    )

    model_version = model_store.create_model_version(
        db,
        ModelVersionCreateRepo(
            created_by_id=user.id,
            mlflow_version=None,
            check_status=(
                "OK" if model_create.config.framework != "torch" else None
            ),
            mlflow_model_name=mlflow_name,
            model_id=model.id,
            name=model_create.config.name,
            config=model_create.config,
            description=model_create.model_version_description,
        ),
    )
    task = None
    if model_create.config.framework == "torch":
        assert model_create.config.dataset.target_columns[
            0
        ].out_module, "missing out_module in target_column"

        task = _check_model(
            model_version=ModelVersion.from_orm(model_version),
            user=user,
            task_control=get_task_control(),
        )

    assert model_version.id in [version.id for version in model.versions]
    model = Model.from_orm(model)
    if return_async_task:
        return model, task
    return model


def get_models(db: Session, query: ModelsQuery, current_user: UserEntity):
    """Get all registered models"""
    models, total = model_store.get_paginated(
        db,
        created_by_id=current_user.id,
        per_page=query.per_page,
        page=query.page,
        q=query.q,
    )
    parsed_models = [Model.from_orm(model) for model in models]
    return parsed_models, total


def get_model_options() -> List[options.ComponentOption]:
    """
    Gets all component (featurizers and layer) options supported by the system,
    along with metadata about each.
    """
    return options.options_manager.options


class PredictRequest(ApiBaseModel):
    """Payload of a prediction request"""

    user_id: int
    model_version_id: int
    model_input: Any


def get_model_prediction(
    db: Session, request: PredictRequest
) -> Dict[str, List[Any]]:
    """(Slowly) Loads a model version and apply it to a sample input

    .. warning::
        Avoid using this function until it is adapted to run from ray worker.

    Args:
        db: Session with the database
        request: prediction request data

    Returns:
        model output

    Raises:
        ModelVersionNotFound: If the version from request.model_version_id
        is not in the database
        InvalidDataframe: If the dataframe fails one or more checks
    """
    modelversion = ModelVersion.from_orm(
        model_store.get_model_version(db, request.model_version_id)
    )
    if not modelversion:
        raise ModelVersionNotFound()
    if not modelversion.mlflow_version:
        raise ModelVersionNotTrained()

    return predict(
        mlflow_model_name=modelversion.mlflow_model_name,
        mlflow_model_version=modelversion.mlflow_version,
        spec=modelversion.config,
        input_=request.model_input,
    )


def get_model(db: Session, user: UserEntity, model_id: int) -> Model:
    """Gets a single model"""
    modeldb = model_store.get(db, model_id)
    if not modeldb or modeldb.created_by_id != user.id:
        raise ModelNotFound()
    model = Model.from_orm(modeldb)
    return model


def delete_model(db: Session, user: UserEntity, model_id: int) -> Model:
    """Deletes a model and all it's artifacts"""
    model = model_store.get(db, model_id)
    if not model or model.created_by_id != user.id:
        raise ModelNotFound()
    client = mlflow.tracking.MlflowClient()
    client.delete_registered_model(model.mlflow_name)
    parsed_model = Model.from_orm(model)
    model_store.delete_by_id(db, model_id)
    return parsed_model


async def update_model_version(
    db: Session,
    user: UserEntity,
    version_id: int,
    version_update: ModelVersionUpdate,
    model_id: int | None = None,
    return_async_task=False,
) -> ModelVersion:
    """
    Updates a single model version.

    Model versions are only open for updates when a model training check points to a failure.

    Args:
        db: Session with the database.
        user: User making the request.
        version_id: Model version id to update.
        version_update: Model version update data.
        model_id (optional): Model id. If provided, the model version must belong to this model.
        return_async_task (optional): If True, returns a tuple with the model version and the
            async task that will update the model version. If False, returns only the model version.
            Defaults to False.
    Returns:
        Updated model version.
    Raises:
        ModelVersionNotFound: If the version from request.model_version_id
        is not in the database.
        ModelVersionNotUpdatable: If the model version is not updatable.
    """
    modelversion = model_store.get_model_version(db, version_id)
    if not modelversion:
        raise ModelVersionNotFound("Model version not found")
    if model_id is not None and modelversion.model_id != model_id:
        raise ModelVersionNotFound(
            f"Model version not found in model with id {model_id}"
        )

    if modelversion.check_status != "FAILED":
        raise ModelVersionNotUpdatable(
            "Only allowed to update model versions that have failed the check status"
        )
    modelversion = model_store.update_model_version(
        db,
        version_id,
        {"check_status": "RUNNING", "config": version_update.config.dict()},
    )

    task = _check_model(
        model_version=ModelVersion.from_orm(modelversion),
        user=user,
        task_control=get_task_control(),
    )

    if return_async_task:
        return ModelVersion.from_orm(modelversion), task
    return ModelVersion.from_orm(modelversion)
