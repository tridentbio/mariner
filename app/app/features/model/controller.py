from fastapi.datastructures import UploadFile
from sqlalchemy.orm.session import Session
from app.core.mlflowapi import create_registered_model, create_tracking_client, create_deployment_with_endpoint
from app.features.model.crud import repo
from app.features.model.deployments.crud import repo as deployment_repo
from app.features.model.deployments.schema import Deployment, DeploymentCreate
from app.features.model.schema.model import Model, ModelCreate, ModelCreateRepo, ModelsQuery
from app.features.user.schema import User


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
        endpoint_name=endpoint_name,
        model_uri=f'models:/{model_name}/{latest_version}'
    )
    deployment_entity = deployment_repo.create(db, obj_in=deployment_create)
    return Deployment.from_orm(deployment_entity)

    
def get_models(
    db: Session,
    query: ModelsQuery
):
    print(query)
    models, total = repo.get_paginated(db, per_page=query.per_page, page=query.page)
    return models, total

