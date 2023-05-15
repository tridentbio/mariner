"""
Deployment entity and core relations
"""
from enum import Enum as PyEnum
from typing import List

from sqlalchemy import Boolean, Column, Enum, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql.functions import current_timestamp
from sqlalchemy.sql.schema import ForeignKey
from sqlalchemy.sql.sqltypes import DateTime

from mariner.db.base_class import Base
from mariner.entities.model import ModelVersion
from mariner.entities.user import User


class DeploymentStatus(str, PyEnum):
    STOPPED = "stopped"
    ACTIVE = "active"
    IDLE = "idle"
    STARTING = "starting"


class ShareStrategy(str, PyEnum):
    PUBLIC = "public"
    PRIVATE = "private"


class RateLimitUnit(str, PyEnum):
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    MONTH = "month"


class Predictions(Base):
    """Entity mapping to predictions of deployment."""

    id = Column(Integer, primary_key=True, index=True)
    deployment_id = Column(
        Integer, ForeignKey("deployment.id", ondelete="CASCADE"), nullable=False
    )

    user_id = Column(Integer, ForeignKey("user.id"), nullable=True)
    user = relationship(User)

    created_at = Column(DateTime, default=current_timestamp())


class SharePermissions(Base):
    """Entity mapping to share permissions of deployment."""

    id = Column(Integer, primary_key=True, index=True)
    deployment_id = Column(
        Integer, ForeignKey("deployment.id", ondelete="CASCADE"), nullable=False
    )
    deployment = relationship("Deployment", back_populates="share_permissions")

    user_id = Column(Integer, ForeignKey("user.id"), nullable=True)
    user = relationship(
        User,
    )

    organization = Column(String, nullable=True)

    @classmethod
    def build(cls, users_id: List[int] = [], organizations: List[str] = []) -> list:
        """Build a list of SharePermission from users_id and organizations.

        Args:
            users_id (List[int], optional): List of users id. Defaults to [].
            organizations (List[str], optional): List of organizations. Defaults to [].

        Returns:
            List of SharePermissions
        """
        share_permissions = []
        if len(users_id):
            share_permissions += [cls(user_id=id) for id in users_id]
        if len(organizations):
            share_permissions += [
                cls(organization=org_alias) for org_alias in organizations
            ]
        return share_permissions


class Deployment(Base):
    """Entity mapping to deployment created by users."""

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, unique=True)
    readme = Column(String, nullable=True)
    share_url = Column(String, nullable=True)
    status = Column(
        Enum(DeploymentStatus), default=DeploymentStatus.STOPPED, index=True
    )
    prediction_rate_limit_unit = Column(Enum(RateLimitUnit), nullable=True)
    prediction_rate_limit_value = Column(Integer, nullable=True)
    show_training_data = Column(Boolean, default=False)

    share_strategy = Column(Enum(ShareStrategy), default=ShareStrategy.PRIVATE)
    share_permissions = relationship(
        "SharePermissions", back_populates="deployment", cascade="all,delete"
    )

    model_version_id = Column(
        Integer, ForeignKey("modelversion.id", ondelete="CASCADE"), nullable=False
    )
    model_version = relationship(ModelVersion)

    created_by_id = Column(Integer, ForeignKey("user.id"))
    created_by = relationship(User)

    created_at = Column(DateTime, default=current_timestamp())
    updated_at = Column(DateTime, server_default=current_timestamp())
    deleted_at = Column(DateTime, nullable=True)
