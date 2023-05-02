"""
Module containing deployment service functions.
"""
from typing import List, Tuple

from sqlalchemy.orm.session import Session

from mariner.core.security import (
    decode_deployment_url_token,
    generate_deployment_signed_url,
)
from mariner.entities.deployment import Deployment as DeploymentEntity, ShareStrategy
from mariner.entities.user import User
from mariner.exceptions import DeploymentNotFound, NotCreatorOwner
from mariner.schemas.deployment_schemas import (
    Deployment,
    DeploymentBase,
    DeploymentCreateRepo,
    DeploymentsQuery,
    DeploymentUpdateRepo,
    PermissionCreateRepo,
)
from mariner.stores.deployment_sql import deployment_store


def get_deployments(
    db: Session, current_user: User, query: DeploymentsQuery
) -> Tuple[List[Deployment], int]:
    """Retrieve all deployments that the requester has access to.

    Args:
        db (Session): SQLAlchemy session.
        current_user (User): Current user.
        query (DeploymentsQuery): Deployments query object.

    Returns:
        Tuple[List[Deployment], int]: A tuple containing a list of deployments and the total number of deployments.
    """
    db_data, total = deployment_store.get_many_paginated(db, query, current_user)
    return db_data, total


def create_deployment(db: Session, current_user: User, deployment_input: DeploymentBase) -> Deployment:
    """Create a deployment for a model version.

    Args:
        db (Session): SQLAlchemy session.
        current_user (User): Current user.
        deployment_input (DeploymentBase): Deployment input object.

    Returns:
        Deployment: Created deployment.

    Raises:
        ModelVersionNotFound: If the model version does not exist.
        NotCreatorOwner: If the user is not the creator of the deployment.
    """
    deployment_store.get_model_version(db, deployment_input.model_version_id, current_user.id)

    deployment_create = DeploymentCreateRepo(
        **deployment_input.dict(),
        created_by_id=current_user.id,
    )
    deployment = deployment_store.create(db, deployment_create)

    if deployment_input.share_strategy == ShareStrategy.PUBLIC:
        share_url = generate_deployment_signed_url(deployment.id)
        deployment = deployment_store.update(db, deployment, DeploymentUpdateRepo(share_url=share_url))

    return Deployment.from_orm(deployment)


def update_deployment(
    db: Session, current_user: User, deployment_id: int, deployment_input: DeploymentUpdateRepo
) -> Deployment:
    """Update a deployment.

    Args:
        db (Session): SQLAlchemy session.
        current_user (User): Current user.
        deployment_id (int): Deployment ID.
        deployment_input (DeploymentUpdateRepo): Deployment input object.

    Returns:
        Deployment: Updated deployment.

    Raises:
        ModelVersionNotFound: If the model version does not exist.
        NotCreatorOwner: If the user is not the creator of the deployment.
    """
    deployment_entity: DeploymentEntity = deployment_store.get(db, deployment_id)
    if not deployment_entity:
        raise DeploymentNotFound()
    if deployment_entity.created_by_id != current_user.id:
        raise NotCreatorOwner()

    if (
        deployment_input.share_strategy == ShareStrategy.PUBLIC
        and not deployment_entity.share_url
    ):
        share_url = generate_deployment_signed_url(deployment_entity.id)
        deployment_input.share_url = share_url

    if deployment_input.status != deployment_entity.status:
        ...  # TODO: change service status

    return deployment_store.update(db, db_obj=deployment_entity, obj_in=deployment_input)


def delete_deployment(db: Session, current_user: User, deployment_to_delete_id: int) -> Deployment:
    """Delete a deployment.

    Args:
        db (Session): SQLAlchemy session.
        current_user (User): Current user.
        deployment_to_delete_id (int): Deployment ID to be deleted.

    Returns:
        Deployment: Deleted deployment (with updated status).

    Raises:
        NotCreatorOwner: If the user is not the creator of the deployment.
    """
    deployment = deployment_store.get(db, deployment_to_delete_id)
    if deployment.created_by_id != current_user.id:
        raise NotCreatorOwner()

    return Deployment.from_orm(
        deployment_store.update(db, deployment, DeploymentUpdateRepo(delete=True))
    )


def create_permission(
    db: Session, current_user: User, permission_input: PermissionCreateRepo
) -> Deployment:
    """Create a permission for a deployment.

    Args:
        db (Session): SQLAlchemy session.
        current_user (User): Current user.
        permission_input (PermissionCreateRepo): Permission input object.

    Returns:
        Deployment: Deployment with updated permissions.

    Raises:
        NotCreatorOwner: If the user is not the creator of the deployment.
    """
    deployment_entity: DeploymentEntity = deployment_store.get(db, permission_input.deployment_id)
    if current_user.id != deployment_entity.created_by_id:
        raise NotCreatorOwner("Unauthorized.")
    deployment_store.create_permission(db, permission_input)

    return Deployment.from_orm(deployment_store.get(db, permission_input.deployment_id))


def delete_permission(
    db: Session, current_user: User, permission_input: PermissionCreateRepo
) -> Deployment:
    """Delete a permission for a deployment.

    Args:
        db (Session): SQLAlchemy session.
        current_user (User): Current user.
        permission_input (PermissionCreateRepo): Permission input object.

    Returns:
        Deployment: Deployment with updated permissions.

    Raises:
        NotCreatorOwner: If the user is not the creator of the deployment.
    """
    deployment_entity: DeploymentEntity = deployment_store.get(db, permission_input.deployment_id)
    if current_user.id != deployment_entity.created_by_id:
        raise NotCreatorOwner("Unauthorized.")

    deployment_store.delete_permission(db, permission_input)

    return Deployment.from_orm(deployment_store.get(db, permission_input.deployment_id))


def get_public_deployment(db: Session, token):
    """Get a public deployment.
    Token should be a jwt with the sub field set to the deployment id.
    Deployment needs to have the share_strategy set to public.
    
    Args:
        db (Session): SQLAlchemy session.
        token (str): JWT token.
        
    Returns:
        Deployment
        
    Raises:
        DeploymentNotFound: If the deployment does not exist.
        PermissionError: If the deployment is not public.
    """
    payload = decode_deployment_url_token(token)

    deployment = deployment_store.get(db, payload.sub)
    if not deployment:
        raise DeploymentNotFound()

    elif not deployment.share_strategy == ShareStrategy.PUBLIC:
        raise PermissionError()

    return deployment
