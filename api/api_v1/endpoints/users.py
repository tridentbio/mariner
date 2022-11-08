from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import mariner.users as controller
from api import deps
from mariner.entities.user import User
from mariner.schemas import user_schemas as schemas
from mariner.stores.user_sql import user_store

router = APIRouter()


@router.get(
    "/",
    response_model=List[schemas.User],
    dependencies=[Depends(deps.get_current_active_superuser)],
)
def read_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Retrieve users.
    """
    users = controller.get_users(db, skip=skip, limit=limit)
    return users


@router.put("/me", response_model=schemas.User)
def update_user_me(
    db: Session = Depends(deps.get_db),
    request: schemas.UserUpdate = Depends(schemas.UserUpdate),
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Update own user.
    """
    user = controller.update_user(db, request, current_user)
    return user


@router.get("/me", response_model=schemas.User)
def read_user_me(
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Get current user.
    """
    return current_user


@router.put(
    "/{user_id}",
    response_model=schemas.User,
    dependencies=[Depends(deps.get_current_active_superuser)],
)
def update_user(
    *,
    db: Session = Depends(deps.get_db),
    user_id: int,
    user_in: schemas.UserUpdate,
) -> Any:
    """
    Update a user.
    """
    user = user_store.get(db, id=user_id)
    if not user:
        raise HTTPException(
            status_code=404,
            detail="The user with this username does not exist in the system",
        )
    user = user_store.update(db, db_obj=user, obj_in=user_in)
    return user
