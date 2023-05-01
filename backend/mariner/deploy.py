"""
Deploy service
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
    """Retrieve all deploys that requester has access to."""
    return deploy_store.get_many_paginated(db, query, current_user)


def create_deploy(db: Session, current_user: User, deploy_input: DeployBase) -> Deploy:
    """Create a deploy for a model version.

    Args:
        db (Session): SQLAlchemy session
        current_user (User): Current user
        deploy_input (DeployBase): Deploy input

    Returns:
        Deploy: Created deploy

    Raises:
        ModelVersionNotFound: If model version does not exist
        NotCreatorOwner: If user is not the creator of the deploy
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
        db (Session): SQLAlchemy session
        current_user (User): Current user
        deploy_id (int): Deploy id
        deploy_input (DeployUpdateRepo): Deploy input

    Returns:
        Deploy: Updated deploy

    Raises:
        ModelVersionNotFound: If model version does not exist
        NotCreatorOwner: If user is not the creator of the deploy
    """
    deploy: DeployEntity = deploy_store.get(db, deploy_id)
    if not deploy:
        raise DeployNotFound()
    if deploy.created_by_id != current_user.id:
        raise NotCreatorOwner()

    if deploy_input.share_strategy == ShareStrategy.PUBLIC and not deploy.share_url:
        share_url = generate_deploy_signed_url(deploy.id)
        deploy_input.share_url = share_url

    if deploy_input.status != deploy.status:
        ...  # TODO: change service status

    return deploy_store.update(db, db_obj=deploy, obj_in=deploy_input)


def delete_deploy(
    db: Session, current_user: User, deploy_to_delete: DeployUpdateRepo
) -> Deploy:
    deploy = deploy_store.get(db, deploy_to_delete.id)
    if deploy.created_by_id != current_user.id:
        raise NotCreatorOwner()

    return Deploy.from_orm(
        deploy_store.update(db, deploy, DeployUpdateRepo(delete=True))
    )


def create_permission(
    db: Session, current_user: User, permission_input: PermissionCreateRepo
) -> Deploy:
    # TODO: create share permission for deploy
    return Deploy()


def delete_permission(db: Session, current_user: User, permission_id: int) -> Deploy:
    # TODO: delete share permission
    return Deploy()
