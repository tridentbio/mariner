"""
Deploy related DTOs
"""

from typing import List, Optional

from pydantic import root_validator

from mariner.entities.deploy import (
    DeploymentStatus,
    RateLimitUnit,
    ShareStrategy,
)
from mariner.schemas.api import ApiBaseModel, PaginatedApiQuery, utc_datetime


class DeploymentsQuery(PaginatedApiQuery):
    """Query for deploys.

    Attributes:
        name (Optional[str]): name of the deploy
        status (Optional[DeploymentStatus]): status of the deploy
        share_strategy (Optional[ShareStrategy]): share strategy of the deploy
        created_after (Optional[utc_datetime]): created after date
        modelVersionId (Optional[int]): model version id
        created_by_id (Optional[int]): created by id
    """

    name: Optional[str]
    status: Optional[DeploymentStatus]
    share_strategy: Optional[ShareStrategy]
    created_after: Optional[utc_datetime]
    model_version_id: Optional[int]
    created_by_id: Optional[int]


class DeployBase(ApiBaseModel):
    """Deploy base type."""

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
    created_by_user_id: int
    deleted_at: Optional[utc_datetime] = None


class DeployCreateRepo(DeployBase):
    """Deploy to create type.

    Used in the DeployCRUD to create a deploy in the database.
    """


class Deploy(DeployBase):
    """Deploy model type."""

    id: int


class DeployUpdateInput(ApiBaseModel):
    """Payload of a deploy update.


    Attributes:
        name (str): name of the deploy
        readme (str): readme of the deploy
        share_url (str): share url of the deploy
        status (DeploymentStatus): status of the deploy
        model_version_id (int): model version id of the deploy
        share_strategy (ShareStrategy): share strategy of the deploy
        users_id_allowed (List[int]): users id allowed of the deploy
        organizations_allowed (List[str]): organizations allowed of the deploy
        show_training_data (bool): show training data of the deploy
        prediction_rate_limit_value (int): prediction rate limit value of the deploy
        prediction_rate_limit_unit (RateLimitUnit): prediction rate limit unit of the deploy

    """

    name: str = None
    readme: str = None
    status: DeploymentStatus = None
    share_strategy: ShareStrategy = None
    users_id_allowed: List[int] = None
    organizations_allowed: List[str] = None
    show_training_data: bool = None
    prediction_rate_limit_value: int = None
    prediction_rate_limit_unit: RateLimitUnit = None


class DeployUpdateRepo(DeployUpdateInput):
    """Deploy to update type.

    Used in the deploy update route.
    """

    share_url: str = None
    delete = False  # when true it updates the deleted_at field


class PermissionBase(ApiBaseModel):
    """Permission base type."""

    deploy_id: int
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
