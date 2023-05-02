"""
Handlers for api/v1/deployments* endpoints
"""
from typing import Any

from fastapi.exceptions import HTTPException
from fastapi.param_functions import Depends
from fastapi.routing import APIRouter
from sqlalchemy.orm.session import Session
from starlette import status

import mariner.deployment as controller
from api import deps
from mariner.entities.user import User
from mariner.exceptions import (
    DeploymentAlreadyExists,
    DeploymentNotFound,
    NotCreatorOwner,
)
from mariner.schemas.api import Paginated
from mariner.schemas.deployment_schemas import (
    Deployment,
    DeploymentBase,
    DeploymentsQuery,
    DeploymentUpdateInput,
    DeploymentUpdateRepo,
    PermissionCreateRepo,
    PermissionDeleteRepo,
)

router = APIRouter()


@router.get("/", response_model=Paginated[Deployment])
def get_deployments(
    query: DeploymentsQuery = Depends(DeploymentsQuery),
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Retrieve deployments owned by requester
    """
    deployments, total = controller.get_deployments(db, current_user, query)
    return Paginated(data=[Deployment.from_orm(d) for d in deployments], total=total)


@router.post("/", response_model=Deployment)
def create_deployment(
    data: DeploymentBase,
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Create a deployment
    """
    try:
        db_deployment = controller.create_deployment(db, current_user, data)

        deployment = Deployment.from_orm(db_deployment)
        return deployment

    except DeploymentNotFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    except DeploymentAlreadyExists:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Deployment name already in use"
        )


@router.put(
    "/{deployment_id}",
    response_model=Deployment,
    dependencies=[Depends(deps.get_current_active_user)],
)
def update_deployment(
    deployment_id: int,
    deployment_input: DeploymentUpdateInput,
    current_user=Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
):
    """
    Update a deployment and process again if it is needed
    """
    try:
        deployment = controller.update_deployment(
            db,
            current_user,
            deployment_id,
            DeploymentUpdateRepo(
                **deployment_input.dict(),
            ),
        )
        return deployment
    except DeploymentNotFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    except NotCreatorOwner:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)


@router.delete("/{deployment_id}", response_model=Deployment)
def delete_deployment(
    deployment_id: int,
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
):
    """Delete a deployment by id"""
    try:
        deployment = controller.delete_deployment(db, current_user, deployment_id)
        return deployment
    except DeploymentNotFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    except NotCreatorOwner:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


@router.post("/create-permission", response_model=Deployment)
def create_permission(
    permission_input: PermissionCreateRepo,
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
):
    """Create a permission for a deployment"""
    try:
        deployment = controller.create_permission(db, current_user, permission_input)
        return deployment
    except DeploymentNotFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    except NotCreatorOwner:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)


@router.post("/delete-permission", response_model=Deployment)
def delete_permission(
    query: PermissionDeleteRepo,
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
):
    """Delete a permission for a deployment"""
    try:
        deployment = controller.delete_permission(db, current_user, query)
        return deployment
    except DeploymentNotFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    except NotCreatorOwner:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)


@router.get("/public/{token}", response_model=Deployment)
def get_public_deployment(
    token: str,
    db: Session = Depends(deps.get_db),
):
    """Get a public deployment by token without authentication"""
    try:
        deployment = controller.get_public_deployment(db, token)
        return deployment
    except DeploymentNotFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    except PermissionError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
