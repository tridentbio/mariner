"""
Handlers for api/v1/deployments* endpoints
"""
from typing import Any, Dict, Union, List

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
    ModelVersionNotFound,
    NotCreatorOwner,
    PredictionLimitReached,
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
    DeploymentManagerComunication
)

router = APIRouter()


@router.get("/", response_model=Paginated[Deployment])
def get_deployments(
    query: DeploymentsQuery = Depends(DeploymentsQuery),
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
) -> Paginated[Deployment]:
    """
    Retrieve deployments owned by requester
    """
    deployments, total = controller.get_deployments(db, current_user, query)
    return Paginated(data=deployments, total=total)

@router.get("/{deployment_id}", response_model=Deployment)
def get_deployment(
    deployment_id: int,
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
) -> Deployment:
    """
    Retrieve a deployment
    """
    try:
        deployment = controller.get_deployment(db, current_user, deployment_id)
        return deployment
    except DeploymentNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Deployment not found"
        )


@router.post("/", response_model=Deployment)
def create_deployment(
    deployment_base: DeploymentBase,
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
) -> Deployment:
    """
    Create a deployment
    """
    try:
        db_deployment = controller.create_deployment(db, current_user, deployment_base)

        deployment = Deployment.from_orm(db_deployment)
        return deployment

    except DeploymentAlreadyExists:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Deployment name already in use",
        )

    except ModelVersionNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Model version not found"
        )

    except NotCreatorOwner:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not creator owner of model version",
        )


@router.put(
    "/{deployment_id}",
    response_model=Deployment,
    dependencies=[Depends(deps.get_current_active_user)],
)
async def update_deployment(
    deployment_id: int,
    deployment_input: DeploymentUpdateInput,
    current_user=Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
):
    """
    Update a deployment and handle the status on ray cluster if needed
    """
    try:
        deployment = await controller.update_deployment(
            db,
            current_user,
            deployment_id,
            DeploymentUpdateRepo(
                **deployment_input.dict(),
            ),
        )
        return deployment
    except DeploymentNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Deployment not found"
        )

    except NotCreatorOwner:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not creator owner of deployment",
        )

    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Started and idle are not implemented yet for a deployment",
        )


@router.delete("/{deployment_id}", response_model=Deployment)
async def delete_deployment(
    deployment_id: int,
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
):
    """Delete a deployment by id"""
    try:
        deployment = await controller.delete_deployment(db, current_user, deployment_id)
        return deployment
    except DeploymentNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Deployment not found"
        )
    except NotCreatorOwner:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not creator owner of deployment",
        )


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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Deployment not found"
        )
    except NotCreatorOwner:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not creator owner of deployment",
        )


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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Deployment not found"
        )
    except NotCreatorOwner:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not creator owner of deployment",
        )


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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Deployment not found."
        )

    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Deployment is not public."
        )


@router.post("/{deployment_id}/predict", response_model=Dict[str, Any])
async def post_make_prediction_deployment(
    deployment_id: int,
    data: Dict[str, Any],
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
):
    """Make a prediction in a deployment instance."""
    try:
        prediction: Dict[str, Any] = await controller.make_prediction(
            db, current_user, deployment_id, data
        )
        return prediction
    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="You don't have permission to make predictions for this deployment.",
        )

    except PredictionLimitReached:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="You have reached the prediction limit for this deployment.",
        )

    except DeploymentNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Deployment not found.",
        )



@router.post(
    "/deployment-manager", 
    response_model=Union[Deployment, List[Deployment]],  
    dependencies=[Depends(deps.assert_trusted_service)]
)
def handle_deployment_manager(
    message: DeploymentManagerComunication,
    db: Session = Depends(deps.get_db),
):
    try:
        if message.first_init:
            return controller.handle_deployment_manager_first_init(db)
        else:
            return controller.update_deployment_status(db, message.deployment_id, message.status)
    
    except DeploymentNotFound:
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Deployment not found.",
        )