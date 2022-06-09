from fastapi.datastructures import UploadFile
from sqlalchemy.orm.session import Session

from app.core.mlflowapi import create_registered_model, create_tracking_client
from app.features.model.crud import repo
from app.features.model.schema.model import Model, ModelCreate


def create_model(
    db: Session,
    model_create: ModelCreate,
    torchmodel: UploadFile,
) -> Model:
    client = create_tracking_client()
    mlflow_reg_model, _ = create_registered_model(
        client,
        model_create.name,
        torchmodel,
        description=model_create.model_description,
        version_description=model_create.model_description,
    )
    model = Model.from_orm(repo.create(db, obj_in=model_create))
    model.set_from_mlflow_model(mlflow_reg_model)
    return model
