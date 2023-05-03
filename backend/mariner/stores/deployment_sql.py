"""
Deployment data layer defining ways to read and write to the deployments collection
"""
from datetime import datetime
from typing import List

from sqlalchemy import or_
from sqlalchemy.orm import Session

from mariner.entities.deployment import (
    Deployment,
    SharePermissions,
    ShareStrategy,
)
from mariner.entities.user import User
from mariner.exceptions import ModelVersionNotFound, NotCreatorOwner
from mariner.entities.model import ModelVersion
from mariner.schemas.model_schemas import ModelVersion as ModelVersionSchema
from mariner.schemas.deployment_schemas import Deployment as DeploymentSchema
from mariner.schemas.deployment_schemas import (
    DeploymentCreateRepo,
    DeploymentsQuery,
    DeploymentUpdateRepo,
    PermissionCreateRepo,
    PermissionDeleteRepo,
)
from mariner.stores.base_sql import CRUDBase

from .model_sql import model_store


class CRUDDeployment(CRUDBase[Deployment, DeploymentCreateRepo, DeploymentUpdateRepo]):
    """CRUD for :any:`Deployment model<mariner.entities.deployment.Deployment>`

    Responsible to handle all communication with the deployment for the Deployment model.
    """

    def get_many_paginated(self, db: Session, query: DeploymentsQuery, user: User):
        """Get many deployment based on the query parameters

        Args:
            db (Session): deployment session
            query (DeploymentsQuery): query parameters

        Returns:
            A tuple with the list of deployment and the total number of deployment
        """
        sql_query = (
            db.query(Deployment, SharePermissions, ModelVersion)
            .join(SharePermissions, Deployment.share_permissions, isouter=True)
            .distinct(Deployment.id)
        )
        sql_query = sql_query.filter(Deployment.deleted_at.is_(None))

        if query.access_mode == "owned":
            sql_query = sql_query.filter(
                Deployment.created_by_id == user.id
            )

        elif query.public_mode == "only":
            sql_query = sql_query.filter(
                Deployment.share_strategy == ShareStrategy.PUBLIC
            )

        else:
            filters = [
                SharePermissions.user_id == user.id,
                SharePermissions.organization == f"@{user.email.split('@')[1]}",
            ]

            if not query.access_mode == "shared":
                filters.append(Deployment.created_by_id == user.id)

            if query.public_mode == "include":
                filters.append(Deployment.share_strategy == "PUBLIC")

            sql_query = sql_query.filter(or_(*filters))

        if query.name:
            sql_query = sql_query.filter(Deployment.name.ilike(f"%{query.name}%"))
        if query.status:
            sql_query = sql_query.filter(Deployment.status == query.status)
        if query.share_strategy:
            sql_query = sql_query.filter(
                Deployment.share_strategy == query.share_strategy
            )
        if query.created_after:
            sql_query = sql_query.filter(Deployment.created_at >= query.created_after)
        if query.model_version_id:
            sql_query = sql_query.filter(
                Deployment.model_version_id == query.model_version_id
            )

        total = sql_query.count()
        sql_query = sql_query.limit(query.per_page).offset(query.page * query.per_page)

        result = sql_query.all()

        deployments: List[DeploymentSchema] = []
        for i, record in enumerate(result):
            deployment, share_permissions, model_version = record

            deployments.append(DeploymentSchema.from_orm(deployment))
            if share_permissions:
                if share_permissions.user_id:
                    deployments[i].users_id_allowed.append(share_permissions.user_id)
                elif share_permissions.organization:
                    deployments[i].organizations_allowed.append(
                        share_permissions.organization
                    )
            deployments[i].model_version = ModelVersionSchema.from_orm(model_version)

        return deployments, total

    def create(self, db: Session, obj_in: DeploymentCreateRepo):
        """Create a new deployment

        Args:
            db (Session): database session
            obj_in (DeploymentCreateRepo): deployment to be created

        Returns:
            Created deployment
        """
        obj_in_dict = obj_in.dict()
        relations_key = ["users_id_allowed", "organizations_allowed"]
        ds_data = {
            k: obj_in_dict[k] for k in obj_in_dict.keys() if k not in relations_key
        }
        db_obj = Deployment(
            **ds_data,
            share_permissions=self.parse_share_permissions(
                users_id=obj_in_dict["users_id_allowed"],
                organizations=obj_in_dict["organizations_allowed"],
            ),
        )

        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
        self,
        db: Session,
        db_obj: Deployment,
        obj_in: DeploymentUpdateRepo,
    ):
        """Update a deployment

        Args:
            db (Session): database session
            db_obj (Deployment): deployment to be updated
            obj_in (DeploymentUpdateRepo): deployment data to be updated

        Returns:
            Updated deployment
        """
        if obj_in.users_id_allowed or obj_in.organizations_allowed:
            share_permissions = self.parse_share_permissions(
                users_id=obj_in.users_id_allowed or [],
                organizations=obj_in.organizations_allowed or [],
            )
            db.query(SharePermissions).filter(
                SharePermissions.deployment_id == db_obj.id,
            ).delete()
            db_obj = db.query(Deployment).filter(Deployment.id == db_obj.id).first()
            db_obj.share_permissions = share_permissions
            db.add(db_obj)
            db.commit()

        del obj_in.organizations_allowed
        del obj_in.users_id_allowed

        super().update(db, db_obj=db_obj, obj_in=obj_in.dict(exclude_none=True))
        return db_obj

    def parse_share_permissions(
        self, users_id: List[int] = [], organizations: List[str] = []
    ):
        """Parse share permissions from users_id and organizations
        to a list of SharePermissions.

        Args:
            users_id (List[int], optional): List of users id. Defaults to [].
            organizations (List[str], optional): List of organizations. Defaults to [].

        Returns:
            List of SharePermissions
        """
        share_permissions = []
        if len(users_id):
            share_permissions += [SharePermissions(user_id=id) for id in users_id]
        if len(organizations):
            share_permissions += [
                SharePermissions(organization=org_alias) for org_alias in organizations
            ]
        return share_permissions

    def create_permission(self, db: Session, obj_in: PermissionCreateRepo):
        """Create a new permission

        Args:
            db (Session): database session
            obj_in (PermissionCreateRepo): permission to be created

        Returns:
            Created permission
        """
        if not isinstance(obj_in, dict):
            obj_in = obj_in.dict()

        db_obj = SharePermissions(**obj_in)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def delete_permission(self, db: Session, obj_in: PermissionDeleteRepo):
        """Delete a permission

        Args:
            db (Session): database session
            obj_in (PermissionDeleteRepo): permission to be deleted

        Returns:
            Deleted permission
        """
        sub_query = (
            SharePermissions.user_id == obj_in.user_id
            if obj_in.user_id
            else SharePermissions.organization == obj_in.organization
        )

        db_obj = (
            db.query(SharePermissions)
            .filter(SharePermissions.deployment_id == obj_in.deployment_id, sub_query)
            .first()
        )
        db.delete(db_obj)
        db.commit()
        return db_obj

    def get_model_version(self, db: Session, model_version_id: int, user_id: int):
        """Safely gets the model version if it is owned by the requester,
        or raises a 404 error.

        Args:
            db (Session): the database session
            model_version_id (int): the ID of the model version to retrieve
            user_id (int): the ID of the user making the request

        Raises:
            ModelVersionNotFound: if the model version is not found
            NotCreatorOwner: if the user is not the owner of the model version

        Returns:
            ModelVersion
        """
        model_version = model_store.get_model_version(db, model_version_id)

        model = model_store.get(db, model_version.model_id)
        if not model:
            raise ModelVersionNotFound("Model not found")
        if model.created_by_id != user_id:
            raise NotCreatorOwner("You are not the owner of this model version")

        return model_version


deployment_store = CRUDDeployment(Deployment)
