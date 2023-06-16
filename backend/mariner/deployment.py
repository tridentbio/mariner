"""
Module containing deployment service functions.
"""
import asyncio
from typing import Any, Dict, List, Tuple

from sqlalchemy.orm.session import Session
from torch import Tensor

from api.websocket import WebSocketMessage, get_websockets_manager
from mariner.core.security import (
    decode_deployment_url_token,
    generate_deployment_signed_url,
)
from mariner.entities.deployment import Deployment as DeploymentEntity
from mariner.entities.deployment import ShareStrategy
from mariner.entities.user import User
from mariner.exceptions import (
    DeploymentNotFound,
    NotCreatorOwner,
    PredictionLimitReached,
)
from mariner.ray_actors.deployments_manager import get_deployments_manager
from mariner.schemas.deployment_schemas import (
    Deployment,
    DeploymentBase,
    DeploymentCreateRepo,
    DeploymentsQuery,
    DeploymentStatus,
    DeploymentUpdateRepo,
    DeploymentWithTrainingData,
    PermissionCreateRepo,
    PredictionCreateRepo,
)
from mariner.stores.deployment_sql import deployment_store
from mariner.tasks import TaskView, get_manager


def get_deployments(
    db: Session, current_user: User, query: DeploymentsQuery
) -> Tuple[List[Deployment], int]:
    """Retrieve a page of deployments that the requester has access to.

    Args:
        db (Session): SQLAlchemy session.
        current_user (User): Current user.
        query (DeploymentsQuery): Deployments query object.

    Returns:
        Tuple[List[Deployment], int]: A tuple containing a list of deployments and the total number of deployments.
    """
    db_data, total = deployment_store.get_many_paginated(db, query, current_user)
    return db_data, total


def get_deployment(
    db: Session, current_user: User, deployment_id: int
) -> DeploymentWithTrainingData:
    """Retrieve a deployment that the requester has access to.

    Args:
        db (Session): SQLAlchemy session.
        current_user (User): Current user.
        deployment_id (int): Deployment ID.

    Returns:
        A deployment with training data object.

    Raises:
        DeploymentNotFound: If the deployment does not exist.
        NotCreatorOwner: If the user is not the creator of the deployment.
    """
    deployment = deployment_store.get_if_has_permission(
        db, deployment_id=deployment_id, user=current_user
    )
    if not deployment:
        raise DeploymentNotFound()
    deployment = DeploymentWithTrainingData.from_orm(deployment)

    if deployment.show_training_data:
        deployment.dataset_summary = deployment_store.get_training_data(db, deployment)

    return deployment


def create_deployment(
    db: Session, current_user: User, deployment_input: DeploymentBase
) -> Deployment:
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
    deployment_store.get_model_version(
        db, deployment_input.model_version_id, current_user.id
    )

    deployment_create = DeploymentCreateRepo(
        **deployment_input.dict(),
        created_by_id=current_user.id,
    )
    deployment = deployment_store.create(db, deployment_create)

    if deployment_input.share_strategy == ShareStrategy.PUBLIC:
        share_url = generate_deployment_signed_url(deployment.id)
        deployment = deployment_store.update(
            db, deployment, DeploymentUpdateRepo(share_url=share_url)
        )

    return Deployment.from_orm(deployment)


async def update_deployment(
    db: Session,
    current_user: User,
    deployment_id: int,
    deployment_input: DeploymentUpdateRepo,
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

    if deployment_input.status and deployment_input.status != deployment_entity.status:
        manager = get_deployments_manager()

        if deployment_input.status == "active":
            await manager.add_deployment.remote(Deployment.from_orm(deployment_entity))
            manager.start_deployment.remote(
                deployment_entity.id, deployment_entity.created_by_id
            )
            del deployment_input.status  # status will by handled asynchronously

        elif deployment_input.status == "stopped":
            manager.stop_deployment.remote(
                deployment_entity.id, deployment_entity.created_by_id
            )
            del deployment_input.status  # status will by handled asynchronously

    return deployment_store.update(
        db, db_obj=deployment_entity, obj_in=deployment_input
    )


async def delete_deployment(
    db: Session, current_user: User, deployment_to_delete_id: int
) -> Deployment:
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

    manager = get_deployments_manager()
    if deployment.id in await manager.get_deployments.remote():
        await manager.stop_deployment.remote(deployment.id, deployment.created_by_id)
        await manager.remove_deployment.remote(deployment.id)

    return deployment_store.update(db, deployment, DeploymentUpdateRepo.delete())


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
    deployment_entity: DeploymentEntity = deployment_store.get(
        db, permission_input.deployment_id
    )
    if not deployment_entity:
        raise DeploymentNotFound()
    elif current_user.id != deployment_entity.created_by_id:
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
    deployment_entity: DeploymentEntity = deployment_store.get(
        db, permission_input.deployment_id
    )
    if not deployment_entity:
        raise DeploymentNotFound()
    elif current_user.id != deployment_entity.created_by_id:
        raise NotCreatorOwner("Unauthorized.")

    deployment_store.delete_permission(db, permission_input)

    return Deployment.from_orm(deployment_store.get(db, permission_input.deployment_id))


def get_public_deployment(db: Session, token: str):
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

    deployment = DeploymentWithTrainingData.from_orm(deployment)

    if deployment.show_training_data:
        deployment.dataset_summary = deployment_store.get_training_data(db, deployment)

    return deployment


async def make_prediction(
    db: Session, current_user: User, deployment_id: int, data: Dict[str, Any]
):
    """Make a prediction and track it.

    Args:
        db: SQLAlchemy session.
        current_user: Current user.
        deployment_id: Deployment ID.
        data: Data to be predicted (any json).

    Returns:
        Prediction: Json with predictions for each model version target column.

    Raises:
        PermissionError: If the user does not have access to the deployment.
        PredictionLimitReached: If the user reached the prediction limit.
        DeploymentNotFound: If the deployment is not running.
    """
    manager = get_deployments_manager()
    if not await manager.is_deployment_running.remote(deployment_id):
        raise DeploymentNotFound()

    deployment = deployment_store.get_if_has_permission(db, deployment_id, current_user)
    if not deployment:
        raise PermissionError()

    prediction_count = deployment_store.get_predictions_count(
        db, deployment, current_user
    )
    if prediction_count >= deployment.prediction_rate_limit_value:
        raise PredictionLimitReached()

    prediction: Dict[str, Tensor] = await manager.make_prediction.remote(
        deployment_id, data
    )

    deployment_store.create_prediction_entry(
        db,
        PredictionCreateRepo(
            deployment_id=deployment_id,
            user_id=current_user.id,
        ),
    )

    for column, result in prediction.items():
        assert isinstance(result, Tensor), "Result must be a Tensor"
        serialized_result = result.tolist()
        prediction[column] = (
            serialized_result
            if isinstance(serialized_result, list)
            else [serialized_result]
        )

    return prediction


async def make_prediction_public(db: Session, deployment_id: int, data: Dict[str, Any]):
    """Make a prediction for a public deployment.

    Args:
        db: SQLAlchemy session.
        deployment_id: Deployment ID.
        data: Data to be predicted (any json).

    Returns:
        Prediction: Json with predictions for each model version target column.
    """
    manager = get_deployments_manager()
    if not await manager.is_deployment_running.remote(deployment_id):
        raise DeploymentNotFound()

    deployment = deployment_store.get(db, deployment_id)
    if not deployment or not deployment.share_strategy == ShareStrategy.PUBLIC:
        raise PermissionError("Deployment is not public.")

    prediction_count = deployment_store.get_predictions_count(db, deployment)
    if prediction_count >= deployment.prediction_rate_limit_value:
        raise PredictionLimitReached()

    prediction: Dict[str, Tensor] = await manager.make_prediction.remote(
        deployment_id, data
    )

    deployment_store.create_prediction_entry(
        db,
        PredictionCreateRepo(deployment_id=deployment_id),
    )

    for column, result in prediction.items():
        assert isinstance(result, Tensor), "Result must be a Tensor"
        serialized_result = result.tolist()
        prediction[column] = (
            serialized_result
            if isinstance(serialized_result, list)
            else [serialized_result]
        )

    return prediction


def notify_users_about_status_update(deployment: Deployment):
    """Notify the user using websocket about the deployment status updated.

    This function will start the broadcast on websocket and add it as a task
    to the task manager to finish asynchoronously.

    Args:
        deployment: Deployment object.
    """
    manager = get_manager("deployment")
    task = asyncio.ensure_future(
        get_websockets_manager().broadcast(
            WebSocketMessage(
                type="update-deployment",
                data={
                    "deploymentId": deployment.id,
                    "status": (
                        deployment.status
                        if isinstance(deployment.status, str)
                        else deployment.status.value
                    ),
                },
            ),
            public=deployment.share_strategy == ShareStrategy.PUBLIC,
        )
    )
    manager.add_new_task(
        TaskView(
            id=deployment.id,
            task=task,
            user_id=deployment.created_by_id,
        )
    )


async def update_deployment_status(
    db: Session, deployment_id: int, status: DeploymentStatus
):
    """Update the deployment status on database and notify the user about it.
    To be used by the deployment manager.

    Args:
        db: database session.
        deployment_id,
        status: new status.

    Returns:
        Updated deployment.
    """
    deployment = deployment_store.get(db, deployment_id)
    if not deployment:
        return

    deployment = deployment_store.update(
        db, deployment, DeploymentUpdateRepo(status=status)
    )
    deployment = Deployment.from_orm(deployment)

    notify_users_about_status_update(deployment)
    return deployment
