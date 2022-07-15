from typing import List, Optional, Union

from mlflow.entities.model_registry.registered_model import RegisteredModel
from pydantic.main import BaseModel
from app.features.dataset.schema import Dataset

from app.features.model.schema.configs import ModelConfig
from app.features.user.schema import User
from app.schemas.api import ApiBaseModel, PaginatedApiQuery


class ModelDeployment(ApiBaseModel):
    pass


class ModelVersion(ApiBaseModel):
    model_name: str
    model_version: str
    config: ModelConfig

    def load_from_mlflowapi(self):
        from app.core.mlflowapi import get_model_version

        version = get_model_version(self.model_name, self.model_version)
        return version


class Model(ApiBaseModel):
    name: str
    description: Optional[str] = None
    created_by_id: int
    created_by: Optional[User] = None
    dataset_id: int
    dataset: Optional[Dataset] = None
    versions: List[ModelVersion]

    _loaded: Optional[RegisteredModel] = None

    def get_model_uri(self, version: Optional[Union[str, int]] = None):
        if not self._loaded:
            self._loaded = self.load_from_mlflow()
        if not version:
            version = self.versions[-1].model_version
        return f"models:/{self.name}/{version}"

    def set_from_registered_model(self, regmodel: RegisteredModel):
        self.description = regmodel.description

    def load_from_mlflow(self):
        if self._loaded:
            return self._loaded

        from app.core.mlflowapi import get_registry_model

        registered_model = get_registry_model(self.name)
        self.set_from_registered_model(registered_model)
        self._loaded = registered_model
        return registered_model


class ModelsQuery(PaginatedApiQuery):
    pass


class ModelCreateRepo(BaseModel):
    dataset_id: int
    name: str
    created_by_id: int


class ModelUpdateRepo(Model):
    pass


class ModelCreate(ApiBaseModel):
    name: str
    model_description: Optional[str] = None
    model_version_description: Optional[str] = None
    config: ModelConfig


class ModelVersionCreateRepo(BaseModel):
    model_name: str
    model_version: str
    config: ModelConfig
