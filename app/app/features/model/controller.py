import importlib
from fastapi.datastructures import UploadFile
from sqlalchemy.orm.session import Session
from app.core.mlflowapi import create_registered_model, create_tracking_client, create_deployment_with_endpoint
from app.features.model import generate
from app.features.model.crud import repo
from app.features.model.deployments.crud import repo as deployment_repo
from app.features.model.deployments.schema import Deployment, DeploymentCreate
from app.features.model.schema.configs import ModelOptions
from app.features.model.schema import layers_schema
from app.features.model.schema.model import Model, ModelCreate, ModelCreateRepo, ModelsQuery
from app.features.user.model import User


def create_model(
    db: Session,
    model_create: ModelCreate,
    torchmodel: UploadFile,
) -> Model:
    client = create_tracking_client()
    mlflow_reg_model, version = create_registered_model(
        client,
        model_create.name,
        torchmodel,
        description=model_create.model_description,
        version_description=model_create.model_version_description,
    )
    model = Model.from_orm(
        repo.create(
            db,
            obj_in=ModelCreateRepo(
                name=model_create.name,
                created_by_id=model_create.created_by_id)))
    assert version is not None
    model.set_from_mlflow_model(
        mlflow_reg_model,
        versions=[version] if version else []
    )
    return model


def create_model_deployment(
    db: Session,
    deployment_create: DeploymentCreate,
    user: User
) -> Deployment:
    model_name, latest_version = deployment_create.model_name, deployment_create.model_version
    endpoint_name = f'{user.id}/{model_name}/{latest_version}'
    create_deployment_with_endpoint(
        deployment_name=endpoint_name,
        model_uri=f'models:/{model_name}/{latest_version}'
    )
    deployment_entity = deployment_repo.create(db, obj_in=deployment_create)
    return Deployment.from_orm(deployment_entity)

    
def get_models(
    db: Session,
    query: ModelsQuery,
    current_user: User
):
    models, total = repo.get_paginated(db, created_by_id=current_user.id, per_page=query.per_page, page=query.page)
    models = [ Model.from_orm(model) for model in models ]
    for model in models:
        model.load_from_mlflow()
    return models, total

def get_model_options(
) -> ModelOptions:
    layers = []
    featurizers = []
    layer_types = [l.name for l in generate.layers]
    featurizer_types = [f.name for f in generate.featurizers]
    for class_name in dir(layers_schema):
        if class_name.endswith('ArgsTemplate'):
            cls = getattr(layers_schema, class_name)
            instance = cls()
            print(cls)
            if instance.type in layer_types:
                layers.append(cls())
            if instance.type in featurizer_types:
                featurizers.append(cls())
    return ModelOptions(layers=layers, featurizers=featurizers)
