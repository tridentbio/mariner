"""
Models service
"""
import logging
import traceback
from typing import Any, Dict, List, Literal, Tuple
from uuid import uuid4

import lightning.pytorch as pl
import mlflow
import mlflow.exceptions
import pandas as pd
import torch
from sqlalchemy.orm.session import Session
from torch_geometric.loader import DataLoader

from fleet.model_builder import options
from fleet.model_builder.dataset import CustomDataset
from fleet.model_builder.model import CustomModel
from fleet.model_builder.schemas import (
    ComponentOption,
    DatasetConfig,
    SmileDataType,
)
from mariner.core import mlflowapi
from mariner.entities.user import User as UserEntity
from mariner.exceptions import (
    DatasetNotFound,
    ModelNameAlreadyUsed,
    ModelNotFound,
    ModelVersionNotFound,
)
from mariner.exceptions.model_exceptions import (
    InvalidDataframe,
    ModelVersionNotTrained,
)
from mariner.ray_actors.model_check_actor import ModelCheckActor
from mariner.schemas.api import ApiBaseModel
from mariner.schemas.model_schemas import (
    ForwardCheck,
    Model,
    ModelCreate,
    ModelCreateRepo,
    ModelFeaturesAndTarget,
    ModelSchema,
    ModelsQuery,
    ModelVersion,
    ModelVersionCreateRepo,
)
from mariner.stores.dataset_sql import dataset_store
from mariner.stores.model_sql import model_store
from mariner.validation.functions import is_valid_smiles_series

LOG = logging.getLogger(__file__)
LOG.setLevel(logging.INFO)


def get_model_and_dataloader(
    db: Session, config: ModelSchema
) -> tuple[pl.LightningModule, DataLoader]:
    """Gets the CustomModel and the DataLoader needed to train
    a model

    Args:
        db: connection to the database
        config: model config

    Returns:
        tuple[pl.LightningModule, DataLoader]
    """
    dataset_entity = dataset_store.get_by_name(db, config.dataset.name)
    assert dataset_entity, f"No dataset found with name {config.dataset.name}"
    df = dataset_entity.get_dataframe()
    dataset = CustomDataset(data=df, config=config)
    dataloader = DataLoader(dataset, batch_size=1)
    model = CustomModel(config)
    return model, dataloader


async def check_model_step_exception(db: Session, config: ModelSchema) -> ForwardCheck:
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
        dataset = dataset_store.get_by_name(db, config.dataset.name)
        actor = ModelCheckActor.remote()
        output = await actor.check_model_steps.remote(dataset=dataset, config=config)
        return ForwardCheck(output=output)
    # Don't catch any specific exception to properly get the traceback
    except:  # noqa E722 pylint: disable=W0702
        lines = traceback.format_exc()
        return ForwardCheck(stack_trace=lines)


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


def get_model_options() -> List[ComponentOption]:
    return options.get_model_options()


class PredictRequest(ApiBaseModel):
    """Payload of a prediction request"""

    user_id: int
    model_version_id: int
    model_input: Any


InvalidReason = Literal["invalid-smiles-series"]


def _check_dataframe_conforms_dataset(
    df: pd.DataFrame, dataset_config: DatasetConfig
) -> List[Tuple[str, InvalidReason]]:
    """Checks if dataframe conforms to data types specified in dataset_config

    Args:
        df: dataframe to be checked
        dataset_config: model builder dataset configuration used in model building
    """
    column_configs = dataset_config.feature_columns + dataset_config.target_columns
    result: List[Tuple[str, InvalidReason]] = []
    for column_config in column_configs:
        data_type = column_config.data_type
        if isinstance(data_type, SmileDataType):
            series = df[column_config.name]
            if not is_valid_smiles_series(series):
                result.append((column_config.name, "invalid-smiles-series"))
    return result


def get_model_prediction(
    db: Session, request: PredictRequest
) -> Dict[str, torch.Tensor]:
    """(Slowly) Loads a model version and apply it to a sample input

    Args:
        db: Session with the database
        request: prediction request data

    Returns:
        torch.Tensor: model output

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
    df = pd.DataFrame.from_dict(request.model_input, dtype=float)
    broken_checks = _check_dataframe_conforms_dataset(df, modelversion.config.dataset)
    if len(broken_checks) > 0:
        raise InvalidDataframe(
            f"dataframe failed {len(broken_checks)} checks",
            reasons=[f"{col_name}: {rule}" for col_name, rule in broken_checks],
        )
    dataset = CustomDataset(data=df, config=modelversion.config, target=False)
    dataloader = DataLoader(dataset, batch_size=len(df))
    modelinput = next(iter(dataloader))
    pyfuncmodel = mlflowapi.get_model_by_uri(modelversion.get_mlflow_uri())
    return pyfuncmodel.predict_step(modelinput)


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
