from typing import Any, List

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic.networks import EmailStr
from sqlalchemy.orm import Session

from app import schemas
from app.api import deps
from app.core.config import settings
from app.features.user import controller
from app.features.user.crud import repo
from app.features.user.exceptions import UserAlreadyExists
from app.features.user.model import User
from app.features.user.schema import UserCreate

router = APIRouter()


@router.get("/", response_model=List[schemas.User], dependencies=[Depends(deps.get_current_active_superuser)])
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

@router.post("/", response_model=schemas.User, dependencies=[Depends(deps.get_current_active_superuser)])
def create_user(
    *,
    user_in: UserCreate,
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Create new user.
    """
    try:
        user = controller.create_user(db, user_in=user_in)
        return user
    except (UserAlreadyExists):
        raise HTTPException(
            status_code=400,
            detail="The user with this username already exists in the system.",
        )

# TODO: pass to MVC layer structure
@router.put("/me", response_model=schemas.User)
def update_user_me(
    *,
    db: Session = Depends(deps.get_db),
    password: str = Body(None),
    full_name: str = Body(None),
    email: EmailStr = Body(None),
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Update own user.
    """
    current_user_data = jsonable_encoder(current_user)
    user_in = schemas.UserUpdate(**current_user_data)
    if password is not None:
        user_in.password = password
    if full_name is not None:
        user_in.full_name = full_name
    if email is not None:
        user_in.email = email
    user = repo.update(db, db_obj=current_user, obj_in=user_in)
    return user


# TODO: pass to MVC layer structure
@router.get("/me", response_model=schemas.User)
def read_user_me(
    db: Session = Depends(deps.get_db), # could be in dependencies? maybe should be executed before get_current_active_user
                                        # except that get_current_active_user indirectly depends on the deps.get_db resultself.
                                        # so probably could be removed
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Get current user.
    """
    return current_user


# TODO: pass to MVC layer structure
@router.post("/open", response_model=schemas.User)
def create_user_open(
    *,
    db: Session = Depends(deps.get_db),
    password: str = Body(...),
    email: EmailStr = Body(...),
    full_name: str = Body(None),
) -> Any:
    """
    Create new user without the need to be logged in.
    """
    if not settings.USERS_OPEN_REGISTRATION:
        raise HTTPException(
            status_code=403,
            detail="Open user registration is forbidden on this server",
        )
    user = repo.get_by_email(db, email=email)
    if user:
        raise HTTPException(
            status_code=400,
            detail="The user with this username already exists in the system",
        )
    user_in = schemas.UserCreate(password=password, email=email, full_name=full_name)
    user = repo.create(db, obj_in=user_in)
    return user


# TODO: pass to MVC layer structure
@router.get("/{user_id}", response_model=schemas.User)
def read_user_by_id(
    user_id: int,
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Get a specific user by id.
    """
    user = repo.get(db, id=user_id)
    if user == current_user:
        return user
    if not repo.is_superuser(current_user):
        raise HTTPException(
            status_code=400, detail="The user doesn't have enough privileges"
        )
    return user


# TODO: pass to MVC layer structure
@router.put("/{user_id}", response_model=schemas.User)
def update_user(
    *,
    db: Session = Depends(deps.get_db),
    user_id: int,
    user_in: schemas.UserUpdate,
    current_user: User = Depends(deps.get_current_active_superuser),
) -> Any:
    """
    Update a user.
    """
    user = repo.get(db, id=user_id)
    if not user:
        raise HTTPException(
            status_code=404,
            detail="The user with this username does not exist in the system",
        )
    user = repo.update(db, db_obj=user, obj_in=user_in)
    return user
