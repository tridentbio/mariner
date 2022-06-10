from logging import debug
from fastapi.datastructures import UploadFile
from sqlalchemy.orm.session import Session
from app.core.mlflowapi import create_registered_model, create_tracking_client
from app.features.model.crud import repo
from app.features.model.schema.model import Model, ModelCreate, ModelCreateRepo


def create_model(
    db: Session,
    model_create: ModelCreate,
    torchmodel: UploadFile,
) -> Model:
    print(model_create)
    client = create_tracking_client()
    mlflow_reg_model, version = create_registered_model(
        client,
        model_create.name,
        torchmodel,
        description=model_create.model_description,
        version_description=model_create.model_version_description,
    )
    model = Model.from_orm(repo.create(db, obj_in=ModelCreateRepo(name=model_create.name, created_by_id=model_create.created_by_id)))
    print(model)
    model.set_from_mlflow_model(mlflow_reg_model, versions=[version])
    print(model)
    return model


