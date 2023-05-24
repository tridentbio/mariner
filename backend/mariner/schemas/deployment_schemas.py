"""
Deployment related DTOs
"""

from typing import List, Literal, Optional

from pydantic import root_validator

from mariner.entities.deployment import Deployment as DeploymentEntity
from mariner.entities.deployment import (
    DeploymentStatus,
    RateLimitUnit,
    ShareStrategy,
)
from mariner.schemas.api import ApiBaseModel, PaginatedApiQuery, utc_datetime
from mariner.schemas.dataset_schemas import DatasetSummary
from mariner.schemas.model_schemas import ModelVersion


class DeploymentsQuery(PaginatedApiQuery):
    """Query object for deployments."""

    name: Optional[str] = None
    status: Optional[DeploymentStatus] = None
    share_strategy: Optional[ShareStrategy] = None
    created_after: Optional[utc_datetime] = None
    model_version_id: Optional[int] = None
    public_mode: Literal["include", "exclude", "only"] = "exclude"
    access_mode: Optional[Literal["unset", "owned", "shared"]] = "unset"


class DeploymentBase(ApiBaseModel):
    """Deployment base type."""

    name: str
    readme: str = ""
    share_url: str = None
    status: DeploymentStatus = DeploymentStatus.STOPPED
    model_version_id: int
    share_strategy: ShareStrategy = ShareStrategy.PRIVATE
    users_id_allowed: List[int] = []
    organizations_allowed: List[str] = []
    show_training_data: bool = False
    prediction_rate_limit_value: int
    prediction_rate_limit_unit: RateLimitUnit = RateLimitUnit.MONTH
    deleted_at: Optional[utc_datetime] = None


class DeploymentCreateRepo(DeploymentBase):
    """Deployment to create type.

    Used in the DeploymentCRUD to create a deployment in the database.
    """

    created_by_id: int


class User(ApiBaseModel):
    """User type."""

    id: int
    email: str
    full_name: Optional[str] = None


class Deployment(DeploymentBase):
    """Deployment model type."""

    id: int
    created_by_id: int
    model_version: ModelVersion = None
    users_allowed: List[User] = []
    created_at: utc_datetime
    updated_at: utc_datetime

    @classmethod
    def from_orm(cls, obj):
        deployment = super().from_orm(obj)
        if isinstance(obj, DeploymentEntity):
            if obj.model_version:
                deployment.model_version = ModelVersion.from_orm(obj.model_version)

            if obj.share_permissions:
                deployment.users_allowed = [
                    User.from_orm(permission.user)
                    for permission in obj.share_permissions
                    if permission.user_id
                ]
                deployment.users_id_allowed = list(
                    map(lambda x: x.id, deployment.users_allowed)
                )
                deployment.organizations_allowed = [
                    permission.organization
                    for permission in obj.share_permissions
                    if permission.organization
                ]
        return deployment


class TrainingData(ApiBaseModel):
    """Training Data of deployment Model Version."""

    dataset_summary: DatasetSummary = None


class DeploymentWithTrainingData(Deployment):
    """Deployment model type with stats field."""

    training_data: TrainingData = None


class DeploymentUpdateInput(ApiBaseModel):
    """Payload of a deployment update."""

    name: str = None
    readme: str = None
    status: DeploymentStatus = None
    share_strategy: ShareStrategy = None
    users_id_allowed: List[int] = None
    organizations_allowed: List[str] = None
    show_training_data: bool = None
    prediction_rate_limit_value: int = None
    prediction_rate_limit_unit: RateLimitUnit = None


class DeploymentUpdateRepo(DeploymentUpdateInput):
    """Deployment to update type.

    Used in the deployment update route.
    """

    share_url: str = None
    deleted_at: Optional[utc_datetime] = None

    @classmethod
    def delete(cls):
        """Returns a DeploymentUpdateRepo with delete set to True."""
        return cls(deleted_at=utc_datetime.now())


class PermissionBase(ApiBaseModel):
    """Permission base type.
    Permissions define who can see/use a private deployment.
    You can allow a certain user to see the deployment by its id.
    You can allow a group of users to view the deployment by their
    email suffix, alias organization (e.g.: "@example.com" would
    allow any user that has an email ending in "@example.com" to view
    the content).
    You can only create one user_id or organization permission at a time.
    """

    deployment_id: int
    user_id: int = None
    organization: str = None

    @root_validator
    def check_redundant_fields(cls, values):
        """Checks that only one of user_id or organization is set."""
        if values.get("user_id") and values.get("organization"):
            raise ValueError("Only one of user_id or organization can be set.")
        if not values.get("user_id") and not values.get("organization"):
            raise ValueError("One of user_id or organization must be set.")
        return values


class PermissionCreateRepo(PermissionBase):
    """Permission to create type.

    Used in the PermissionCRUD to create a permission in the database.
    """


class PermissionDeleteRepo(PermissionBase):
    """Permission to delete type.

    Used in the PermissionCRUD to delete a permission in the database.
    """


class PredictionBase(ApiBaseModel):
    """Prediction base type.

    Used to make and track predictions in deployed models.
    Each prediction should be tracked in the database to be possible
    to limit user predictions by deployment configuration.
    """

    deployment_id: int
    user_id: int = None


class PredictionCreateRepo(PredictionBase):
    """Prediction to create type.

    Used in the PredictionCRUD to create a prediction register in the database.
    """


class DeploymentManagerComunication(ApiBaseModel):
    """Schema used by deployment manager to comunicate with backend

    Used to update a specific deployment status or handle first initialization
    of deployment manager ray actor
    """

    first_init = False
    deployment_id: Optional[int] = None
    status: Optional[DeploymentStatus] = None
