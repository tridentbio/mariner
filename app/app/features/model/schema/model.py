from typing import Any, List, Optional, Union

from mlflow.entities.model_registry.registered_model import (
    ModelVersion,
    RegisteredModel,
)
from pydantic.main import BaseModel

from app.features.model.schema.configs import ModelConfig
from app.features.user.schema import User
from app.schemas.api import ApiBaseModel, PaginatedApiQuery


class ModelDeployment(ApiBaseModel):
    pass


class Model(ApiBaseModel):
    name: str
    model_description: Optional[str] = None
    model_version_description: Optional[str] = None
    created_by_id: int
    created_by: Optional[User] = None
    latest_versions: List[Any] = []
    mlflow_model_data_loaded: bool = False

    def get_model_uri(self, version: Optional[Union[str, int]] = None):
        if not self.mlflow_model_data_loaded:
            self.load_from_mlflow()
        if not version:
            # version = self.latest_versions[-1].version
            version = "1"
        return f"models:/{self.name}/{version}"

    def set_from_mlflow_model(
        self, mlflow_reg_model: RegisteredModel, versions: List[ModelVersion]
    ):
        """
        Merge attributes of interest in a mlflow RegisteredModel
        """
        self.model_description = mlflow_reg_model.description
        self.latest_versions = versions
        if len(versions) > 0:
            self.model_version_description = versions[0].description
        self.mlflow_model_data_loaded = True

    def load_from_mlflow(self):
        from app.core.mlflowapi import get_registry_model

        registered_model = get_registry_model(self.name)
        self.set_from_mlflow_model(registered_model, registered_model.latest_versions)


class ModelsQuery(PaginatedApiQuery):
    pass


class ModelCreateRepo(BaseModel):
    name: str
    created_by_id: int


class ModelUpdateRepo(Model):
    pass


class ModelCreate(ApiBaseModel):
    name: str
    model_description: Optional[str] = None
    model_version_description: Optional[str] = None
    config: ModelConfig
