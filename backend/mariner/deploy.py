"""
Deploy service
"""
from typing import List, Tuple

from sqlalchemy.orm.session import Session

from mariner.entities.user import User
from mariner.schemas.deploy_schemas import (
    Deploy,
    DeployCreateRepo,
    DeploymentsQuery,
    DeployUpdateRepo,
    PermissionCreateRepo,
)


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


def create_deploy(
    db: Session, current_user: User, deploy_input: DeployCreateRepo
) -> Deploy:
    # TODO: create deploy
    return Deploy()


def update_deploy(
    db: Session, current_user: User, deploy_id: int, deploy_input: DeployUpdateRepo
) -> Deploy:
    # TODO: update deploy
    return Deploy()


def delete_deploy(
    db: Session, current_user: User, deploy_to_delete: DeployUpdateRepo
) -> Deploy:
    # TODO: delete deploy
    return Deploy()


def create_permission(
    db: Session, current_user: User, permission_input: PermissionCreateRepo
) -> Deploy:
    # TODO: create share permission for deploy
    return Deploy()


def delete_permission(db: Session, current_user: User, permission_id: int) -> Deploy:
    # TODO: delete share permission
    return Deploy()
