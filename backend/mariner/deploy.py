"""deploy.py
Module containing deploy service functions.
"""
from typing import List, Tuple

from sqlalchemy.orm.session import Session

from mariner.core.security import generate_deploy_signed_url
from mariner.entities.deploy import Deploy as DeployEntity
from mariner.entities.deploy import ShareStrategy
from mariner.entities.user import User
from mariner.exceptions import DeployNotFound, NotCreatorOwner
from mariner.schemas.deploy_schemas import (
    Deploy,
    DeployBase,
    DeployCreateRepo,
    DeploymentsQuery,
    DeployUpdateRepo,
    PermissionCreateRepo,
)
from mariner.stores.deploy_sql import deploy_store


def get_deploys(
    db: Session, current_user: User, query: DeploymentsQuery
) -> Tuple[List[Deploy], int]:
    """Retrieve all deploys that the requester has access to.

    Args:
        db (Session): SQLAlchemy session.
        current_user (User): Current user.
        query (DeploymentsQuery): Deployments query object.

    Returns:
        Tuple[List[Deploy], int]: A tuple containing a list of deploys and the total number of deploys.
    """
    db_data, total = deploy_store.get_many_paginated(db, query, current_user)
    return db_data, total


def create_deploy(db: Session, current_user: User, deploy_input: DeployBase) -> Deploy:
    """Create a deploy for a model version.

    Args:
        db (Session): SQLAlchemy session.
        current_user (User): Current user.
        deploy_input (DeployBase): Deploy input object.

    Returns:
        Deploy: Created deploy.

    Raises:
        ModelVersionNotFound: If the model version does not exist.
        NotCreatorOwner: If the user is not the creator of the deploy.
    """
    deploy_store.get_model_version(db, deploy_input.model_version_id, current_user.id)

    deploy_create = DeployCreateRepo(
        **deploy_input.dict(),
        created_by_id=current_user.id,
    )
    deploy = deploy_store.create(db, deploy_create)

    if deploy_input.share_strategy == ShareStrategy.PUBLIC:
        share_url = generate_deploy_signed_url(deploy.id)
        deploy = deploy_store.update(db, deploy, DeployUpdateRepo(share_url=share_url))

    return Deploy.from_orm(deploy)


def update_deploy(
    db: Session, current_user: User, deploy_id: int, deploy_input: DeployUpdateRepo
) -> Deploy:
    """Update a deploy.

    Args:
        db (Session): SQLAlchemy session.
        current_user (User): Current user.
        deploy_id (int): Deploy ID.
        deploy_input (DeployUpdateRepo): Deploy input object.

    Returns:
        Deploy: Updated deploy.

    Raises:
        ModelVersionNotFound: If the model version does not exist.
        NotCreatorOwner: If the user is not the creator of the deploy.
    """
    deploy_entity: DeployEntity = deploy_store.get(db, deploy_id)
    if not deploy_entity:
        raise DeployNotFound()
    if deploy_entity.created_by_id != current_user.id:
        raise NotCreatorOwner()

    if deploy_input.share_strategy == ShareStrategy.PUBLIC and not deploy_entity.share_url:
        share_url = generate_deploy_signed_url(deploy_entity.id)
        deploy_input.share_url = share_url

    if deploy_input.status != deploy_entity.status:
        ...  # TODO: change service status

    return deploy_store.update(db, db_obj=deploy_entity, obj_in=deploy_input)


def delete_deploy(db: Session, current_user: User, deploy_to_delete_id: int) -> Deploy:
    """Delete a deploy.

    Args:
        db (Session): SQLAlchemy session.
        current_user (User): Current user.
        deploy_to_delete_id (int): Deploy ID to be deleted.

    Returns:
        Deploy: Deleted deploy (with updated status).

    Raises:
        NotCreatorOwner: If the user is not the creator of the deploy.
    """
    deploy = deploy_store.get(db, deploy_to_delete_id)
    if deploy.created_by_id != current_user.id:
        raise NotCreatorOwner()

    return Deploy.from_orm(
        deploy_store.update(db, deploy, DeployUpdateRepo(delete=True))
    )


def create_permission(
    db: Session, current_user: User, permission_input: PermissionCreateRepo
) -> Deploy:
    """Create a permission for a deploy.

    Args:
        db (Session): SQLAlchemy session.
        current_user (User): Current user.
        permission_input (PermissionCreateRepo): Permission input object.

    Returns:
        Deploy: Deploy with updated permissions.

    Raises:
        NotCreatorOwner: If the user is not the creator of the deploy.
    """
    deploy_entity: DeployEntity = deploy_store.get(db, permission_input.deploy_id)
    if current_user.id != deploy_entity.created_by_id:
        raise NotCreatorOwner("Unauthorized.")
    deploy_store.create_permission(db, permission_input)

    return Deploy.from_orm(deploy_store.get(db, permission_input.deploy_id))


def delete_permission(
    db: Session, current_user: User, permission_input: PermissionCreateRepo
) -> Deploy:
    """Delete a permission for a deploy.

    Args:
        db (Session): SQLAlchemy session.
        current_user (User): Current user.
        permission_input (PermissionCreateRepo): Permission input object.

    Returns:
        Deploy: Deploy with updated permissions.

    Raises:
        NotCreatorOwner: If the user is not the creator of the deploy.
    """
    deploy_entity: DeployEntity = deploy_store.get(db, permission_input.deploy_id)
    if current_user.id != deploy_entity.created_by_id:
        raise NotCreatorOwner("Unauthorized.")

    deploy_store.delete_permission(db, permission_input)

    return Deploy.from_orm(deploy_store.get(db, permission_input.deploy_id))