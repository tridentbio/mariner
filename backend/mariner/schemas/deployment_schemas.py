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
    """Query object for deployments.

    Attributes:
        name: to filter deployments by name
        status: to filter deployments by status (stopped, running, idle)
        share_strategy:
            to filter deployments by share strategy (private, public)
            "public" will not work if public_mode is set to "exclude"
        public_mode: "include" | "exclude" | "only"
            to include or exclude public deployments from a query
            exclude (default): exclude public deployments on result
            include: include public deployments on result
            only: only return public deployments
        created_after: created after date
        model_version_id: filter deployments by model version id
        created_by_id:
            prioritizes the deployments filter searching only the
            deployments made by the requested user, excluding public
            and shared deployments
    """

    name: Optional[str] = None
    status: Optional[DeploymentStatus] = None
    share_strategy: Optional[ShareStrategy] = None
    created_after: Optional[utc_datetime] = None
    model_version_id: Optional[int] = None
    public_mode: Literal["include", "exclude", "only"] = "exclude"
    created_by_id: Optional[int] = None


class DeploymentBase(ApiBaseModel):
    """Deployment base type.

    Attributes:
    name (str): name of the deployment
    readme (str): readme of the deployment
    share_url (str): signed URL to get the deployment without authentication
        only exists if the deployment share_strategy is public
    status (DeploymentStatus): the current status of the deployment
    model_version_id (int): the ID of the model version associated with the deployment
    share_strategy (ShareStrategy): change share mode of the deployment
        - PUBLIC: anyone on the internet can access the deployment with the share_url
            anyone logged in can see the deployment on the list of deployments
        - PRIVATE: only users with access to the deployment can access it
    user_ids_allowed (List[int]): list of user IDs allowed to get the deployment
    organizations_allowed (List[str]): list of organizations allowed to get the deployment
        organizations are identified by suffix of the users email (e.g. @mariner.ai)
    show_training_data (bool): if True, the training data will be shown on the deployment page
    rate_limit_value (int): number of requests allowed in the rate_limit_unit
    rate_limit_unit (RateLimitUnit): unit of time to limit the number of requests
        e.g.:
            if rate_limit_value is 10 and rate_limit_unit is RateLimitUnit.DAY,
            the deployment will only allow 10 requests per day
    deleted_at (Optional[utc_datetime]): date of deletion of the deployment, if it is deleted
    """

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
    """Payload of a deployment update."""

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
