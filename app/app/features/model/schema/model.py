from typing import Any, List, Optional

from mlflow.entities.model_registry.registered_model import RegisteredModel, ModelVersion
from pydantic.main import BaseModel

from app.features.user.schema import User
from app.schemas.api import ApiBaseModel, PaginatedApiQuery


ModelVersion = Any

class ModelDeployment(ApiBaseModel):
    pass



class Model(ApiBaseModel):
    name: str
    model_description: Optional[str] = None
    model_version_description: Optional[str] = None
    created_by_id: int
    created_by: Optional[User] = None
    latest_versions: List[ModelVersion] = []

    def set_from_mlflow_model(self, mlflow_reg_model: RegisteredModel, versions: List[ModelVersion]):
        """
        Merge attributes of interest in a mlflow RegisteredModel
        """
        self.model_description = mlflow_reg_model.description
        self.latest_versions = versions
        if len(versions) > 0:
            self.model_version_description = versions[0].description


class ModelsQuery(PaginatedApiQuery):
    pass


class ModelCreateRepo(BaseModel):
    name: str
    created_by_id: int
    pass


class ModelUpdateRepo(Model):
    pass


class ModelCreate(ApiBaseModel):
    name: str
    model_description: Optional[str] = None
    model_version_description: Optional[str] = None
    created_by_id: int

