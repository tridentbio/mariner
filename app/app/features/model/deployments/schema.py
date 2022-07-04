from pydantic.main import BaseModel

from app.features.model.schema.model import Model
from app.features.user.schema import User
from app.schemas.api import ApiBaseModel


class Deployment(ApiBaseModel):
    name: str
    model_name: str
    created_by_id: int
    created_by: User
    model: Model


class DeploymentCreate(ApiBaseModel):
    name: str
    model_name: str
    model_version: str


class DeploymentUpdate(DeploymentCreate):
    pass


class DeploymentCreateRepo(BaseModel):
    name: str
    model_name: str
    model_version: str
    created_by_id: int
