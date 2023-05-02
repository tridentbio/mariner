"""
Deployment related DTOs
"""

from typing import List, Literal, Optional

from pydantic import root_validator

from mariner.entities.deployment import (
    DeploymentStatus,
    RateLimitUnit,
    ShareStrategy,
)
from mariner.schemas.api import ApiBaseModel, PaginatedApiQuery, utc_datetime


class DeploymentsQuery(PaginatedApiQuery):
    """Query for deployments.

    Attributes:
        name (Optional[str]): name of the deployment
        status (Optional[DeploymentStatus]): status of the deployment
        share_strategy (Optional[ShareStrategy]): share strategy of the deployment
        created_after (Optional[utc_datetime]): created after date
        modelVersionId (Optional[int]): model version id
        created_by_id (Optional[int]): created by id
    """

    name: Optional[str] = None
    status: Optional[DeploymentStatus] = None
    share_strategy: Optional[ShareStrategy] = None
    created_after: Optional[utc_datetime] = None
    model_version_id: Optional[int] = None
    public_mode: Literal["include", "exclude", "only"] = "exclude"
    created_by_id: Optional[int] = None


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
    rate_limit_value: int
    rate_limit_unit: RateLimitUnit = RateLimitUnit.MONTH
    deleted_at: Optional[utc_datetime] = None


class DeploymentCreateRepo(DeploymentBase):
    """Deployment to create type.

    Used in the DeploymentCRUD to create a deployment in the database.
    """

    created_by_id: int


class Deployment(DeploymentBase):
    """Deployment model type."""

    id: int
    created_by_id: int

    @root_validator
    def check_fields(cls, values):
        """Checks that only one of user_id or organization is set."""
        return values


class DeploymentUpdateInput(ApiBaseModel):
    """Payload of a deployment update.


    Attributes:
        name (str): name of the deployment
        readme (str): readme of the deployment
        share_url (str): share url of the deployment
        status (DeploymentStatus): status of the deployment
        model_version_id (int): model version id of the deployment
        share_strategy (ShareStrategy): share strategy of the deployment
        users_id_allowed (List[int]): users id allowed of the deployment
        organizations_allowed (List[str]): organizations allowed of the deployment
        show_training_data (bool): show training data of the deployment
        rate_limit_value (int): prediction rate limit value of the deployment
        rate_limit_unit (RateLimitUnit): prediction rate limit unit of the deployment

    """

    name: str = None
    readme: str = None
    status: DeploymentStatus = None
    share_strategy: ShareStrategy = None
    users_id_allowed: List[int] = None
    organizations_allowed: List[str] = None
    show_training_data: bool = None
    rate_limit_value: int = None
    rate_limit_unit: RateLimitUnit = None


class DeploymentUpdateRepo(DeploymentUpdateInput):
    """Deployment to update type.

    Used in the deployment update route.
    """

    share_url: str = None
    delete = False  # when true it updates the deleted_at field
    deleted_at: Optional[utc_datetime] = None


class PermissionBase(ApiBaseModel):
    """Permission base type."""

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
