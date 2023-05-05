"""
Model related DTOs
"""
from datetime import datetime
from typing import Any, List, Literal, Optional, Union

from mlflow.entities.model_registry.registered_model import RegisteredModel
from pydantic import BaseModel

from fleet.base_schemas import BaseFleetModelSpec
from fleet.model_builder.schemas import LossType
from fleet.torch_.schemas import TorchModelSpec
from mariner.schemas.api import ApiBaseModel, PaginatedApiQuery, utc_datetime
from mariner.schemas.dataset_schemas import Dataset
from mariner.schemas.user_schemas import User


class ModelVersion(ApiBaseModel):
    """A version of model architecture in the registry"""

    id: int
    model_id: int
    name: str
    description: Union[str, None] = None
    mlflow_version: Union[str, None] = None
    mlflow_model_name: str
    config: TorchModelSpec
    created_at: utc_datetime
    updated_at: datetime

    def get_mlflow_uri(self):
        """Gets the mlflow model uri to load this model"""
        return f"models:/{self.mlflow_model_name}/{self.mlflow_version}"


class ModelFeaturesAndTarget(ApiBaseModel):
    """
    Represent a column used by the model
    """

    model_id: Union[int, None] = None
    column_name: str
    column_type: Literal["feature", "target"]


class Model(ApiBaseModel):
    """
    Represents a group of model versions in the registry
    """

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
        """Gets a model version with version string ``version``

        Args:
            version: string or number of the version

        Returns:
            ModelVersion if it's found
        """
        for version_item in self.versions:
            if version_item.name == version:
                return version_item
        return None

    def get_model_uri(self, version: Optional[Union[str, int]] = None):
        """Gets the model uri of a version

        The the model uri of the version of this model with version equals ``version``
        or the latest one

        Args:
            version: string or int version representation
        """
        if not version:
            version = self.versions[-1].mlflow_version
        return f"models:/{self.mlflow_name}/{version}"


class ModelsQuery(PaginatedApiQuery):
    """
    Query to get models

    ``q`` can be a description or a name
    """

    q: Optional[str] = None
    dataset_id: Optional[int] = None


class ModelCreateRepo(BaseModel):
    """
    Model insertion object
    """

    dataset_id: int
    name: str
    created_by_id: int
    columns: List[ModelFeaturesAndTarget]
    mlflow_name: str
    description: Union[str, None] = None


class ModelCreate(ApiBaseModel):
    """
    Model create for the user
    """

    name: str
    model_description: Optional[str] = None
    model_version_description: Optional[str] = None
    config: TorchModelSpec


class ModelVersionCreateRepo(BaseModel):
    """
    Model version insertion object
    """

    mlflow_version: Union[str, None] = None
    description: Union[str, None] = None
    mlflow_model_name: str
    model_id: int
    name: str
    config: TorchModelSpec


class ModelVersionUpdateRepo(BaseModel):
    """
    Model version update object
    """

    mlflow_version: str


class LossOption(ApiBaseModel):
    """
    Option of loss function
    """

    key: LossType
    label: str


class TrainingCheckRequest(ApiBaseModel):
    """Request to a request to check if a model version fitting/training works"""

    model_spec: BaseFleetModelSpec


class TrainingCheckResponse(ApiBaseModel):
    """Response to a request to check if a model version fitting/training works"""

    stack_trace: Optional[str] = None
    output: Optional[Any] = None
