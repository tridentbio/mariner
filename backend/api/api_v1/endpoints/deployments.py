"""
Handlers for api/v1/deployments* endpoints.
"""

from typing import Any, Dict, List, Union

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
    DeploymentManagerComunication,
    DeploymentsQuery,
    DeploymentUpdateInput,
    DeploymentUpdateRepo,
    DeploymentWithTrainingData,
    PermissionCreateRepo,
    PermissionDeleteRepo,
)

router = APIRouter()


@router.get("/", response_model=Paginated[Deployment])
def get_deployments(
    query: DeploymentsQuery = Depends(DeploymentsQuery),
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
) -> Paginated[Deployment]:
    """
    Retrieve deployments that the requester has access to.

    Returns:
        Paginated[Deployment]: paginated list of deployments.
    """
    deployments, total = controller.get_deployments(db, current_user, query)
    return Paginated(data=deployments, total=total)


@router.get("/{deployment_id}", response_model=DeploymentWithTrainingData)
def get_deployment(
    deployment_id: int,
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
) -> DeploymentWithTrainingData:
    """
    Retrieve a deployment by id.
    Includes the training data when Deployment.display_training_data is True.

    Args:
        deployment_id.
        current_user: user must be authenticated.
        db: database session.

    Returns:
        DeploymentWithTrainingData:
            deployment including the dataset summary from training data.
            if available.

    Raises:
        HTTPException(status_code=404): if deployment is not found.
    """
    try:
        deployment = controller.get_deployment(db, current_user, deployment_id)
        return deployment
    except DeploymentNotFound as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Deployment not found"
        ) from e


@router.post("/", response_model=Deployment)
def create_deployment(
    deployment_base: DeploymentBase,
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
) -> Deployment:
    """
    Create a deployment

    Args:
        deployment_base.
        current_user: user must be authenticated.
        db: database session.

    Returns:
        Deployment: the created deployment.

    Raises:
        HTTPException(status_code=409): if deployment name already exists.
        HTTPException(status_code=404): if model version does not exist.
        HTTPException(status_code=401): if user is not the creator owner of the model version.
    """
    try:
        db_deployment = controller.create_deployment(db, current_user, deployment_base)

        deployment = Deployment.from_orm(db_deployment)
        return deployment

    except DeploymentAlreadyExists as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Deployment name already in use",
        ) from e

    except ModelVersionNotFound as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Model version not found"
        ) from e

    except NotCreatorOwner as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not creator owner of model version",
        ) from e


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
    Update a deployment and handle the status on ray cluster if needed.
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
    except DeploymentNotFound as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Deployment not found"
        ) from e

    except NotCreatorOwner as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not creator owner of deployment",
        ) from e

    except NotImplementedError as e:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Started and idle are not implemented yet for a deployment",
        ) from e


@router.delete("/{deployment_id}", response_model=Deployment)
async def delete_deployment(
    deployment_id: int,
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
):
    """Delete a deployment by id by setting a value to deleted_at.

    Args:
        deployment_id.
        current_user: user must be authenticated.
        db: database session.

    Returns:
        Deployment: the deleted deployment.

    Raises:
        HTTPException(status_code=404): if deployment is not found.
        HTTPException(status_code=401): if user is not the creator owner of the deployment.
    """
    try:
        deployment = await controller.delete_deployment(db, current_user, deployment_id)
        return deployment
    except DeploymentNotFound as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Deployment not found"
        ) from e
    except NotCreatorOwner as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not creator owner of deployment",
        ) from e


@router.post("/create-permission", response_model=Deployment)
def create_permission(
    permission_input: PermissionCreateRepo,
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
):
    """Create a permission for a deployment.
    Used to create a new single share permission for a deployment.
    This permission can be for a user or an organization.

    Args:
        permission_input:
            deployment_id
            user_id
            organization
        current_user: user must be authenticated.
        db: database session.

    Returns:
        Deployment: the deployment with the new permission.

    Raises:
        HTTPException(status_code=404): if deployment is not found.
        HTTPException(status_code=404): if user is not the creator owner of the deployment.
    """
    try:
        deployment = controller.create_permission(db, current_user, permission_input)
        return deployment
    except DeploymentNotFound as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Deployment not found"
        ) from e
    except NotCreatorOwner as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not creator owner of deployment",
        ) from e


@router.post("/delete-permission", response_model=Deployment)
def delete_permission(
    query: PermissionDeleteRepo,
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
):
    """Delete a permission for a deployment.
    Used to delete a single share permission for a deployment.

    Args:
        query:
            deployment_id.
            user_id.
            organization.
        current_user: user must be authenticated.
        db: database session.

    Returns:
        Deployment: the deployment with the deleted permission.

    Raises:
        HTTPException(status_code=404): if deployment is not found.
        HTTPException(status_code=404): if user is not the creator owner of the deployment.
    """
    try:
        deployment = controller.delete_permission(db, current_user, query)
        return deployment
    except DeploymentNotFound as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Deployment not found"
        ) from e
    except NotCreatorOwner as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not creator owner of deployment",
        ) from e


@router.get("/public/{token}", response_model=DeploymentWithTrainingData)
def get_public_deployment(
    token: str,
    db: Session = Depends(deps.get_db),
):
    """Get a public deployment by token without authentication.
    Includes the training data when Deployment.display_training_data is True.

    Args:
        token:
            JWT token that is generated when a deployment have theshare_permission
            value set to public.
            This token must be signed by DEPLOYMENT_URL_SIGNATURE_SECRET_KEY.
            variable and must have the following payload:
                sub: some public deployment id.
        db: database session.

    Returns:
        DeploymentWithTrainingData: the public deployment.

    Raises:
        HTTPException(status_code=404): if deployment is not found.
        HTTPException(status_code=401): if deployment is not public.
    """
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
    """Make a prediction in a deployment instance.

    Prediction flow:
    1. Check if deployment exists and user has access.
    2. Check if user is able to make a new prediction without
    reach prediction limit configuration.
    3. Make prediction.
    4. Track prediction.
    5. Return prediction.

    Args:
        deployment_id.
        data (any json): data to make prediction.
        current_user: user must be authenticated.
        db: database session.

    Returns:
        prediction result.

    Raises:
        HTTPException(status_code=404): if deployment is not found.
        HTTPException(status_code=401): if user is not the creator owner of the deployment.
        HTTPException(status_code=429): if user has reached the prediction limit.
    """
    try:
        prediction: Dict[str, Any] = await controller.make_prediction(
            db, current_user, deployment_id, data
        )
        return prediction
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="You don't have permission to make predictions for this deployment.",
        ) from e

    except PredictionLimitReached as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="You have reached the prediction limit for this deployment.",
        ) from e

    except DeploymentNotFound as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Deployment instance not running.",
        ) from e


@router.post("/{deployment_id}/predict-public", response_model=Dict[str, Any])
async def post_make_prediction_deployment_public(
    deployment_id: int,
    data: Dict[str, Any],
    db: Session = Depends(deps.get_db),
):
    """Make a prediction in a public deployment instance.
    Follow same rules as post_make_prediction_deployment but track the predictions
    generically.
    The prediction rate limit rules will be applyed to the deployment instead of
    the user.

    Args:
        deployment_id.
        data (any json): data to make prediction.
        db: database session.

    Returns:
        prediction result.

    Raises:
        HTTPException(status_code=404): if deployment is not found.
        HTTPException(status_code=401): if user is not the creator owner of the deployment.
        HTTPException(status_code=429): if user has reached the prediction limit.
    """
    try:
        prediction: Dict[str, Any] = await controller.make_prediction_public(
            db, deployment_id, data
        )
        return prediction

    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="You don't have permission to make predictions for this deployment.",
        ) from e

    except PredictionLimitReached as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="You have reached the prediction limit for this deployment.",
        ) from e

    except DeploymentNotFound as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Deployment instance not running.",
        ) from e


@router.post(
    "/deployment-manager",
    response_model=Union[Deployment, List[Deployment]],
    dependencies=[Depends(deps.assert_trusted_service)],
)
async def handle_deployment_manager(
    message: DeploymentManagerComunication,
    db: Session = Depends(deps.get_db),
):
    """Handle messages from Deployment Manager.

    Deployment Manager uses this endpoint to update the deployment status on database
    and notify the user about it.

    Args:
        message: message from Deployment Manager.
        db: database session.

    Returns:
        Deployment: the deployment with the updated status.

    Raises:
        HTTPException(status_code=404): if deployment is not found.
    """
    try:
        return await controller.update_deployment_status(
            db, message.deployment_id, message.status
        )

    except DeploymentNotFound as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Deployment not found.",
        ) from e
