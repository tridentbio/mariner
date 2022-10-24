from datetime import datetime
from typing import Dict, List, Literal, Optional, Union

from mlflow.entities.model_registry.registered_model import RegisteredModel
from pydantic.main import BaseModel

from mariner.schemas.api import ApiBaseModel, PaginatedApiQuery, utc_datetime
from mariner.schemas.dataset_schemas import Dataset
from mariner.schemas.user_schemas import User
from model_builder.layers_schema import FeaturizersArgsType, LayersArgsType
from model_builder.model import CustomModel
from model_builder.schemas import ModelSchema


class ModelDeployment(ApiBaseModel):
    pass


class ModelVersion(ApiBaseModel):
    id: int
    model_id: int
    name: str
    mlflow_version: str
    mlflow_model_name: str
    config: ModelSchema
    created_at: utc_datetime
    updated_at: datetime

    def build_torch_model(self):
        return CustomModel(self.config)

    def get_mlflow_uri(self):
        return f"models:/{self.mlflow_model_name}/{self.mlflow_version}"

    def load_from_mlflowapi(self):
        from mariner.core.mlflowapi import get_model_version

        version = get_model_version(self.mlflow_model_name, self.mlflow_version)
        return version


class ModelFeaturesAndTarget(ApiBaseModel):
    model_id: int
    column_name: str
    column_type: Literal["feature", "target"]


class Model(ApiBaseModel):
    id: int
    name: str
    mlflow_name: str
    description: Optional[str] = None
    created_by_id: int
    created_by: Optional[User] = None
    dataset_id: Optional[int] = None
    dataset: Optional[Dataset] = None
    versions: List[ModelVersion]
    columns: List[ModelFeaturesAndTarget]
    created_at: datetime
    updated_at: datetime

    _loaded: Optional[RegisteredModel] = None

    def get_version(self, version: Union[str, int]) -> Optional[ModelVersion]:
        for v in self.versions:
            if v.name == version:
                return v
        return None

    def get_model_uri(self, version: Optional[Union[str, int]] = None):
        if not self._loaded:
            self._loaded = self.load_from_mlflow()
        if not version:
            version = self.versions[-1].mlflow_version
        return f"models:/{self.mlflow_name}/{version}"

    def set_from_registered_model(self, regmodel: RegisteredModel):
        self.description = regmodel.description

    def load_from_mlflow(self):
        if self._loaded:
            return self._loaded

        from mariner.core.mlflowapi import get_registry_model

        registered_model = get_registry_model(self.mlflow_name)
        self.set_from_registered_model(registered_model)
        self._loaded = registered_model
        return registered_model


class ModelsQuery(PaginatedApiQuery):
    q: Optional[str] = None
    dataset_id: Optional[int] = None


class ModelCreateRepo(BaseModel):
    dataset_id: int
    name: str
    created_by_id: int
    columns: List[ModelFeaturesAndTarget]
    mlflow_name: str


class ModelUpdateRepo(Model):
    pass


class ModelCreate(ApiBaseModel):
    name: str
    model_description: Optional[str] = None
    model_version_description: Optional[str] = None
    config: ModelSchema


class ModelVersionCreateRepo(BaseModel):
    mlflow_version: str
    mlflow_model_name: str
    model_id: int
    name: str
    config: ModelSchema


class ModelVersionUpdateRepo(BaseModel):
    mlflow_version: str


class ComponentAnnotation(ApiBaseModel):
    """
    Gives extra information about the layer/featurizer

    """

    docs_link: Optional[str]
    docs: Optional[str]
    output_type: Optional[str]
    class_path: str
    type: Literal["featurizer", "layer"]


class ComponentOption(ComponentAnnotation):
    """
    Describes an option to be used in the ModelSchema.layers or ModelSchema.featurizers
    """

    component: Union[LayersArgsType, FeaturizersArgsType]


ModelOptions = List[ComponentOption]
