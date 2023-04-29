"""
Deploy service
"""
from typing import List, Tuple

from sqlalchemy.orm.session import Session

from mariner.entities.user import User
from mariner.entities.deploy import ShareStrategy, Deploy as DeployEntity
from mariner.schemas.deploy_schemas import (
    Deploy,
    DeployBase,
    DeployCreateRepo,
    DeploymentsQuery,
    DeployUpdateRepo,
    PermissionCreateRepo,
)
from mariner.stores.deploy_sql import deploy_store
from mariner.stores.model_sql import model_store
from mariner.exceptions import NotCreatorOwner, DeployNotFound
from mariner.core.security import generate_deploy_signed_url


def get_all_deploys(
    db: Session, current_user: User, query: DeploymentsQuery
) -> Tuple[List[Deploy], int]:
    # TODO: return all deploys shared with user
    return [], 0


def get_my_deploys(
    db: Session, current_user: User, query: DeploymentsQuery
) -> Tuple[List[Deploy], int]:
    # TODO: return user's deploys
    return [], 0


def get_my_deploy_by_id(db: Session, current_user: User, deploy_id: int) -> Deploy:
    # TODO: return user's deploy by id
    return Deploy()


def get_public_deploy_by_token(db: Session, token: str) -> Deploy:
    # TODO: return public deploy by token
    return Deploy()


def create_deploy(
    db: Session, current_user: User, deploy_input: DeployBase
) -> Deploy:
    # it checks if the model version exists and belongs to the user
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
    deploy: DeployEntity = deploy_store.get(db, deploy_id)
    if not deploy: raise DeployNotFound()
    if deploy.created_by_id != current_user.id: raise NotCreatorOwner()
    
    if deploy_input.share_strategy == ShareStrategy.PUBLIC and not deploy.share_url:
        share_url = generate_deploy_signed_url(deploy.id)
        deploy_input.share_url = share_url
    
    if deploy_input.status != deploy.status:
        ... # TODO: change service status
        
    return deploy_store.update(db, db_obj=deploy, obj_in=deploy_input)


def delete_deploy(
    db: Session, current_user: User, deploy_to_delete: DeployUpdateRepo
) -> Deploy:
    deploy = deploy_store.get(db, deploy_to_delete.id)
    if deploy.created_by_id != current_user.id: raise NotCreatorOwner()
    
    return Deploy.from_orm(deploy_store.update(db, deploy, DeployUpdateRepo(delete=True)))


def create_permission(
    db: Session, current_user: User, permission_input: PermissionCreateRepo
) -> Deploy:
    # TODO: create share permission for deploy
    return Deploy()


def delete_permission(db: Session, current_user: User, permission_id: int) -> Deploy:
    # TODO: delete share permission
    return Deploy()
