"""
Deploy data layer defining ways to read and write to the deploys collection
"""
from typing import List

from sqlalchemy import or_
from sqlalchemy.orm import Session

from mariner.entities.deploy import Deploy, SharePermissions
from mariner.entities.user import User
from mariner.exceptions import ModelVersionNotFound, NotCreatorOwner
from mariner.schemas.deploy_schemas import (
    DeployCreateRepo,
    DeploymentsQuery,
    DeployUpdateRepo,
    PermissionCreateRepo,
    PermissionDeleteRepo,
)
from mariner.stores.base_sql import CRUDBase

from .model_sql import model_store


class CRUDDeploy(CRUDBase[Deploy, DeployCreateRepo, DeployUpdateRepo]):
    """CRUD for :any:`Deploy model<mariner.entities.deploy.Deploy>`

    Responsible to handle all communication with the deploy for the Deploy model.
    """

    def get_many_paginated(self, db: Session, query: DeploymentsQuery, user: User):
        """Get many deploy based on the query parameters

        Args:
            db (Session): deploy session
            query (DeployQuery): query parameters

        Returns:
            A tuple with the list of deploy and the total number of deploy
        """
        sql_query = db.query(Deploy).join(SharePermissions, Deploy.share_permissions)
        sql_query.filter(Deploy.deleted_at.is_(None))

        # filtering for accessible deploys
        if query.created_by_id:
            sql_query.filter(Deploy.created_by_id == query.created_by_id)

        elif query.public_mode == "only":
            sql_query.filter(Deploy.share_strategy == "public")

        else:
            filters = [
                Deploy.created_by_id == user.id,
                SharePermissions.user_id == user.id,
                SharePermissions.organization == f"@{user.email.split('@')[1]}",
            ]

            if query.public_mode == "include":
                filters.append(Deploy.share_strategy == "public")

            sql_query = sql_query.filter(or_(*filters))

        # filtering query
        if query.name:
            sql_query = sql_query.filter(Deploy.name.ilike(f"%{query.name}%"))
        if query.status:
            sql_query = sql_query.filter(Deploy.status == query.status)
        if query.share_strategy:
            sql_query = sql_query.filter(Deploy.share_strategy == query.share_strategy)
        if query.created_after:
            sql_query = sql_query.filter(Deploy.created_at >= query.created_after)
        if query.model_version_id:
            sql_query = sql_query.filter(
                Deploy.model_version_id == query.model_version_id
            )
        total = sql_query.count()
        sql_query = sql_query.limit(query.per_page).offset(query.page * query.per_page)
        result = sql_query.all()
        return result, total

    def create(self, db: Session, obj_in: DeployCreateRepo):
        """Create a new deploy

        Args:
            db (Session): database session
            obj_in (DeployCreateRepo): deploy to be created

        Returns:
            Created deploy
        """
        obj_in_dict = obj_in.dict()
        relations_key = ["users_id_allowed", "organizations_allowed"]
        ds_data = {
            k: obj_in_dict[k] for k in obj_in_dict.keys() if k not in relations_key
        }
        db_obj = Deploy(
            **ds_data,
            share_permissions=self.parse_share_permissions(
                ids=obj_in_dict["users_id_allowed"],
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
        db_obj: Deploy,
        obj_in: DeployUpdateRepo,
    ):
        """Update a deploy

        Args:
            db (Session): database session
            db_obj (Deploy): deploy to be updated
            obj_in (DeployUpdateRepo): deploy data to be updated

        Returns:
            Updated deploy
        """
        if obj_in.delete:
            db.delete(db_obj)
            db.commit()
            return db_obj
        else:
            del obj_in.delete

        if obj_in.users_id_allowed or obj_in.organizations_allowed:
            share_permissions = self.parse_share_permissions(
                ids=obj_in.users_id_allowed or [],
                organizations=obj_in.organizations_allowed or [],
            )
            db.query(SharePermissions).filter(
                SharePermissions.deploy_id == db_obj.id,
            ).delete()
            db.commit()
            db_obj = db.query(Deploy).filter(Deploy.id == db_obj.id).first()
            db_obj.share_permissions = share_permissions
            db.add(db_obj)

        del obj_in.organizations_allowed
        del obj_in.users_id_allowed

        super().update(db, db_obj=db_obj, obj_in=obj_in.dict(exclude_none=True))
        return db_obj

    def parse_share_permissions(
        self, ids: List[int] = [], organizations: List[str] = []
    ):
        share_permissions = []
        if len(ids):
            share_permissions += [SharePermissions(user_id=id) for id in ids]
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
            .filter(SharePermissions.deploy_id == obj_in.deploy_id, sub_query)
            .first()
        )
        db.delete(db_obj)
        db.commit()
        return db_obj

    def get_model_version(self, db: Session, model_version_id: int, user_id: int):
        """Ensure that the user is the owner of the model version

        Args:
            db (Session): database session
            model_version_id (int): model version id
            user_id (int): user id

        Raises:
            ValueError: if the user is not the owner of the model version

        Returns:
            ModelVersion: model version
        """
        model_version = model_store.get_model_version(db, model_version_id)

        model = model_store.get(db, model_version.model_id)
        if not model:
            raise ModelVersionNotFound("Model not found")
        if model.created_by_id != user_id:
            raise NotCreatorOwner("You are not the owner of this model version")

        return model_version


deploy_store = CRUDDeploy(Deploy)
