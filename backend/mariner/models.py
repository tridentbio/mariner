"""
Models service
"""
import logging
import traceback
from typing import Any, Dict, List
from uuid import uuid4

import mlflow
import mlflow.exceptions
from sqlalchemy.orm.session import Session

from fleet import options
from fleet.model_functions import predict
from fleet.ray_actors.model_check_actor import ModelCheckActor
from mariner.core import mlflowapi
from mariner.entities.user import User as UserEntity
from mariner.exceptions import (
    DatasetNotFound,
    ModelNameAlreadyUsed,
    ModelNotFound,
    ModelVersionNotFound,
)
from mariner.exceptions.model_exceptions import ModelVersionNotTrained
from mariner.schemas.api import ApiBaseModel
from mariner.schemas.model_schemas import (
    Model,
    ModelCreate,
    ModelCreateRepo,
    ModelFeaturesAndTarget,
    ModelsQuery,
    ModelVersion,
    ModelVersionCreateRepo,
    TrainingCheckRequest,
    TrainingCheckResponse,
)
from mariner.stores.dataset_sql import dataset_store
from mariner.stores.model_sql import model_store

LOG = logging.getLogger(__file__)
LOG.setLevel(logging.INFO)


async def check_model_step_exception(
    db: Session, config: TrainingCheckRequest
) -> TrainingCheckResponse:
    """Checks the steps of a pytorch lightning model built from config.

    Steps are checked before creating the model on the backend, so the user may fix
    the config based on the exceptions raised.

    Args:
        db: Connection to the database, needed to get the dataset.
        config: ModelSchema instance specifying the model.

    Returns:
        An object either with the stack trace or the model output.
    """
    try:
        dataset = dataset_store.get_by_name(db, config.model_spec.dataset.name)
        actor = ModelCheckActor.remote()
        output = await actor.check_model_steps.remote(
            dataset=dataset, config=config.model_spec
        )
        return TrainingCheckResponse(output=output)
    # Don't catch any specific exception to properly get the traceback
    except:  # noqa E722 pylint: disable=W0702
        lines = traceback.format_exc()
        return TrainingCheckResponse(stack_trace=lines)


def create_model(
    db: Session,
    user: UserEntity,
    model_create: ModelCreate,
) -> Model:
    r"""
    Creates a model and a model version if passed with a
    non-existing `model_create.name`, otherwise creates a
    new model version.
    Triggers the creation of a RegisteredModel and a ModelVersion
    in MLFlow API's
    """
    dataset = dataset_store.get_by_name(db, model_create.config.dataset.name)
    if not dataset:
        raise DatasetNotFound()

    client = mlflowapi.create_tracking_client()
    # Handle case where model_create.name refers to existing model
    existingmodel = model_store.get_by_name_from_user(
        db, model_create.name, user_id=user.id
    )
    if existingmodel:
        model_store.create_model_version(
            db,
            ModelVersionCreateRepo(
                mlflow_version=None,
                mlflow_model_name=existingmodel.mlflow_name,
                model_id=existingmodel.id,
                name=model_create.config.name,
                config=model_create.config,
                description=model_create.model_version_description,
            ),
        )
        return Model.from_orm(existingmodel)

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

    if model_create.config.framework == "torch":
        assert model_create.config.dataset.target_columns[
            0
        ].out_module, "missing out_module in target_column"

    model_store.create_model_version(
        db,
        ModelVersionCreateRepo(
            mlflow_version=None,
            mlflow_model_name=mlflow_name,
            model_id=model.id,
            name=model_create.config.name,
            config=model_create.config,
            description=model_create.model_version_description,
        ),
    )

    model = Model.from_orm(model)
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


def get_model_prediction(db: Session, request: PredictRequest) -> Dict[str, List[Any]]:
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
