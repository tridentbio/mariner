"""
Handlers for api/v1/deployments* endpoints
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
    Retrieve deployments that the requester has access to
    
    Args:
        query:
            name: to filter deployments by name
            status: to filter deployments by status (stopped, running, idle)
            share_strategy:
                to filter deployments by share strategy (private, public)
                "public" will not work if public_mode is set to "exclude"
            public_mode: "include" | "exclude" | "only"
                to include or exclude public deployments from a query
                exclude (default): exclude public deployments on result
                include: include public deployments on result
                only: only return public deployments
            created_after: created after date
            model_version_id: filter deployments by model version id
            access_mode: "unset" | "owned" | "shared
                filter by the access the user has to the deployment
                unset (default): do not filter by access mode
                owned: only return deployments owned by the user
                shared: only return deployments shared with the user
        current_user: user must be authenticated
        db: database session
    
    Returns:
        Paginated[Deployment]: paginated list of deployments
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
        deployment_id
        current_user: user must be authenticated
        db: database session
        
    Returns:
        DeploymentWithTrainingData: 
            deployment including the dataset summary from training data
            if available
    
    Raises:
        HTTPException(status_code=404): if deployment is not found
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
    
    Args:
        deployment_base:
            name (str): name of the deployment
            readme (str): readme of the deployment
            share_url (str): signed URL to get the deployment without authentication
                only exists if the deployment share_strategy is public
            status (DeploymentStatus): the current status of the deployment
            model_version_id (int): the ID of the model version associated with the deployment
            share_strategy (ShareStrategy): change share mode of the deployment
                - PUBLIC: anyone on the internet can access the deployment with the share_url
                    anyone logged in can see the deployment on the list of deployments
                - PRIVATE: only users with access to the deployment can access it
            user_ids_allowed (List[int]): list of user IDs allowed to get the deployment
            organizations_allowed (List[str]): list of organizations allowed to get the deployment
                organizations are identified by suffix of the users email (e.g. @mariner.ai)
            show_training_data (bool): if True, the training data will be shown on the deployment page
            prediction_rate_limit_value (int): number of requests allowed in the prediction_rate_limit_unit
            prediction_rate_limit_unit (RateLimitUnit): unit of time to limit the number of requests
                e.g.:
                    if prediction_rate_limit_value is 10 and prediction_rate_limit_unit is RateLimitUnit.DAY,
                    the deployment will only allow 10 requests per day
        current_user: user must be authenticated
        db: database session
    
    Returns:
        Deployment: the created deployment
        
    Raises:
        HTTPException(status_code=409): if deployment name already exists
        HTTPException(status_code=404): if model version does not exist
        HTTPException(status_code=401): if user is not the creator owner of the model version
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
    """Delete a deployment by id by setting a value to deleted_at
    
    Args:
        deployment_id
        current_user: user must be authenticated
        db: database session
        
    Returns:
        Deployment: the deleted deployment
    
    Raises:
        HTTPException(status_code=404): if deployment is not found
        HTTPException(status_code=401): if user is not the creator owner of the deployment
    """
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
    """Create a permission for a deployment
    Used to create a new single share permission for a deployment
    This permission can be for a user or an organization
    
    Args:
        permission_input:
            deployment_id
            user_id
            organization
        current_user: user must be authenticated
        db: database session
    
    Returns:
        Deployment: the deployment with the new permission
    
    Raises:
        HTTPException(status_code=404): if deployment is not found
        HTTPException(status_code=404): if user is not the creator owner of the deployment
    """
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
    """Delete a permission for a deployment
    Used to delete a single share permission for a deployment
    
    Args:
        query:
            deployment_id
            user_id
            organization
        current_user: user must be authenticated
        db: database session
    
    Returns:
        Deployment: the deployment with the deleted permission
        
    Raises:
        HTTPException(status_code=404): if deployment is not found
        HTTPException(status_code=404): if user is not the creator owner of the deployment
    """
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


@router.get("/public/{token}", response_model=DeploymentWithTrainingData)
def get_public_deployment(
    token: str,
    db: Session = Depends(deps.get_db),
):
    """Get a public deployment by token without authentication
    Includes the training data when Deployment.display_training_data is True.
       
    Args:
        token:
            JWT token that is generated when a deployment have theshare_permission 
            value set to public.
            This token must be signed by DEPLOYMENT_URL_SIGNATURE_SECRET_KEY 
            variable and must have the following payload:
                sub: some public deployment id
        db: database session
    
    Returns:
        DeploymentWithTrainingData: the public deployment
        
    Raises:
        HTTPException(status_code=404): if deployment is not found
        HTTPException(status_code=401): if deployment is not public
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
        deployment_id
        data (any json): data to make prediction
        current_user: user must be authenticated
        db: database session
    
    Returns:
        prediction result
    
    Raises:
        HTTPException(status_code=404): if deployment is not found
        HTTPException(status_code=401): if user is not the creator owner of the deployment
        HTTPException(status_code=429): if user has reached the prediction limit
    """
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
            detail="Deployment instance not running.",
        )
    

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
        deployment_id
        data (any json): data to make prediction
        db: database session
    
    Returns:
        prediction result
        
    Raises:
        HTTPException(status_code=404): if deployment is not found
        HTTPException(status_code=401): if user is not the creator owner of the deployment
        HTTPException(status_code=429): if user has reached the prediction limit
    """
    try:
        prediction: Dict[str, Any] = await controller.make_prediction_public(
            db, deployment_id, data
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
            detail="Deployment instance not running.",
        )


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
    Deployment Manager uses this endpoint to:
        - handle_deployment_manager_first_init: 
            Make sure that the deployment status on databaseis the same as the one 
            on the Deployment Manager.
        - update_deployment_status:
            Update the deployment status on database and notify the user about it.
    
    Args:
        message: message from Deployment Manager
        db: database session
    
    Returns:
        Deployment: the deployment with the updated status
    
    Raises:
        HTTPException(status_code=404): if deployment is not found
    """
    try:
        if message.first_init:
            return controller.handle_deployment_manager_first_init(db)
        else:
            return await controller.update_deployment_status(
                db, message.deployment_id, message.status
            )

    except DeploymentNotFound:
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Deployment not found.",
        )
