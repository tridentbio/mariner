"""
Handlers for api/v1/deploy* endpoints
"""
from typing import Any

from fastapi.exceptions import HTTPException
from fastapi.param_functions import Depends
from fastapi.routing import APIRouter
from sqlalchemy.orm.session import Session
from starlette import status

import mariner.deploy as controller
from api import deps
from mariner.entities.user import User
from mariner.exceptions import (
    DeployAlreadyExists,
    DeployNotFound,
    NotCreatorOwner,
)
from mariner.schemas.api import Paginated
from mariner.schemas.deploy_schemas import (
    Deploy,
    DeployBase,
    DeployCreateRepo,
    DeploymentsQuery,
    DeployUpdateInput,
    DeployUpdateRepo,
    PermissionCreateRepo,
)

router = APIRouter()


@router.get("/feed", response_model=Paginated[Deploy])
def get_all_deploys(
    query: DeploymentsQuery = Depends(DeploymentsQuery),
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Retrieve all deploys shared with requester
    """
    deploy, total = controller.get_all_deploys(db, current_user, query)
    return Paginated(data=[DeploymentsQuery.from_orm(ds) for ds in deploy], total=total)


@router.get("/", response_model=Paginated[Deploy])
def get_my_deploys(
    query: DeploymentsQuery = Depends(DeploymentsQuery),
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Retrieve deploys owned by requester
    """
    deploy, total = controller.get_my_deploys(db, current_user, query)
    return Paginated(data=[DeploymentsQuery.from_orm(ds) for ds in deploy], total=total)


@router.get("/{deploy_id}", response_model=Deploy)
def get_my_deploy(
    deploy_id: int,
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Retrieve deploys owned by requester
    """
    dataset = controller.get_my_dataset_by_id(db, current_user, deploy_id)
    return Deploy.from_orm(dataset)


@router.post("/", response_model=Deploy)
async def create_dataset(
    deploy_input: DeployBase,
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Create a dataset
    """
    try:
        payload = DeployCreateRepo(
            **deploy_input.dict(),
        )

        db_deploy = await controller.create_deploy(db, current_user, payload)

        deploy = Deploy.from_orm(db_deploy)
        return deploy

    except DeployNotFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    except DeployAlreadyExists:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Deploy name already in use"
        )


@router.put(
    "/{deploy_id}",
    response_model=Deploy,
    dependencies=[Depends(deps.get_current_active_user)],
)
async def update_deploy(
    deploy_id: int,
    deploy_input: DeployUpdateInput,
    current_user=Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
):
    """
    Update a deploy and process again if it is needed
    """
    try:
        deploy = await controller.update_deploy(
            db,
            current_user,
            deploy_id,
            DeployUpdateRepo(
                **deploy_input.dict(),
            ),
        )
        return deploy
    except DeployNotFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    except NotCreatorOwner:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)


@router.delete("/{deploy_id}", response_model=Deploy)
def delete_deploy(
    deploy_id: int,
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
):
    """Delete a deploy by id"""
    try:
        deploy = controller.delete_deploy(
            db, current_user, DeployUpdateRepo(id=deploy_id, delete=True)
        )
        return deploy
    except DeployNotFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    except NotCreatorOwner:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)


@router.post("/permissions", response_model=Deploy)
def create_permission(
    permission_input: PermissionCreateRepo,
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
):
    """Create a permission for a deploy"""
    try:
        deploy = controller.create_permission(db, current_user, permission_input)
        return deploy
    except DeployNotFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    except NotCreatorOwner:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)


@router.delete("/permissions/{permission_id}", response_model=Deploy)
def delete_permission(
    permission_id: int,
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
):
    """Delete a permission for a deploy"""
    try:
        deploy = controller.delete_permission(db, current_user, permission_id)
        return deploy
    except DeployNotFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    except NotCreatorOwner:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
