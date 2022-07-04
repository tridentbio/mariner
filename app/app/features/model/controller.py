from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from sqlalchemy.orm.session import Session

from app.core import mlflowapi
from app.features.model import generate
from app.features.model.builder import CustomModel
from app.features.model.crud import repo
from app.features.model.deployments.crud import repo as deployment_repo
from app.features.model.deployments.schema import (
    Deployment,
    DeploymentCreate,
    DeploymentCreateRepo,
)
from app.features.model.exceptions import ModelNotFound
from app.features.model.schema import layers_schema
from app.features.model.schema.configs import ModelOptions
from app.features.model.schema.model import (
    Model,
    ModelCreate,
    ModelCreateRepo,
    ModelsQuery,
)
from app.features.user.model import User as UserEntity
from app.schemas.api import ApiBaseModel


def create_model(
    db: Session,
    user: UserEntity,
    model_create: ModelCreate,
) -> Model:
    client = mlflowapi.create_tracking_client()
    torchmodel = CustomModel(model_create.config)
    mlflow_reg_model, version = mlflowapi.create_registered_model(
        client,
        model_create.name,
        torchmodel,
        description=model_create.model_description,
        version_description=model_create.model_version_description,
    )
    model = Model.from_orm(
        repo.create(
            db,
            obj_in=ModelCreateRepo(name=model_create.name, created_by_id=user.id),
        )
    )
    assert version is not None
    model.set_from_mlflow_model(mlflow_reg_model, versions=[version] if version else [])
    return model


def create_model_deployment(
    db: Session, deployment_create: DeploymentCreate, user: UserEntity
) -> Deployment:
    model_name, latest_version = (
        deployment_create.model_name,
        deployment_create.model_version,
    )
    endpoint_name = f"{user.id}-{model_name}-{latest_version}"
    mlflowapi.create_deployment_with_endpoint(
        deployment_name=endpoint_name,
        model_uri=f"models:/{model_name}/{latest_version}",
    )
    deployment_entity = deployment_repo.create(
        db,
        obj_in=DeploymentCreateRepo(
            created_by_id=user.id,
            name=deployment_create.name,
            model_name=deployment_create.model_name,
            model_version=deployment_create.model_version,
        ),
    )
    return Deployment.from_orm(deployment_entity)


def get_models(db: Session, query: ModelsQuery, current_user: UserEntity):
    models, total = repo.get_paginated(
        db, created_by_id=current_user.id, per_page=query.per_page, page=query.page
    )
    models = [Model.from_orm(model) for model in models]
    for model in models:
        model.load_from_mlflow()
    return models, total


def get_model_options() -> ModelOptions:
    layers = []
    featurizers = []
    layer_types = [layer.name for layer in generate.layers]
    featurizer_types = [f.name for f in generate.featurizers]
    for class_name in dir(layers_schema):
        if class_name.endswith("ArgsTemplate"):
            cls = getattr(layers_schema, class_name)
            instance = cls()
            if instance.type in layer_types:
                layers.append(cls())
            if instance.type in featurizer_types:
                featurizers.append(cls())
    return ModelOptions(layers=layers, featurizers=featurizers)


ModelOutput = Union[np.ndarray, pd.Series, pd.DataFrame, List]


class PredictRequest(ApiBaseModel):
    user_id: int
    model_name: str
    version: Optional[Union[str, int]]
    model_input: Any


def get_model_prediction(db: Session, request: PredictRequest) -> ModelOutput:
    model = repo.get_by_name_from_user(db, request.user_id, request.model_name)
    if not model:
        raise ModelNotFound()
    model = Model.from_orm(model)
    pyfuncmodel = mlflowapi.get_model(model, request.version)
    return pyfuncmodel.predict(pd.DataFrame.from_dict(request.model_input))
