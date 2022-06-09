from typing import List, Optional

from mlflow.entities.model_registry.registered_model import RegisteredModel

from app.features.user.schema import User
from app.schemas.api import ApiBaseModel


class ModelVersion(ApiBaseModel):
    version: int


class Model(ApiBaseModel):
    name: str
    created_by_id = int
    created_by = Optional[User]
    latest_versions: List[ModelVersion] = []

    def set_from_mlflow_model(self, mlflow_reg_model: RegisteredModel):
        """
        Merge attributes of interest in a mlflow RegisteredModel
        """


class ModelCreateRepo(Model):
    pass


class ModelUpdateRepo(Model):
    pass


class ModelCreate(ModelCreateRepo):
    model_description: Optional[str]
    model_version_description: Optional[str]
