from datetime import timedelta
from typing import Optional

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from mariner.core import github, security
from mariner.core.config import settings
from mariner.db.session import SessionLocal
from mariner.entities.user import User as UserEntity
from mariner.exceptions import (
    InvalidOAuthState,
    UserNotActive,
    UserNotFound,
    UserNotSuperUser,
)
from mariner.exceptions.user_exceptions import UserEmailNotAllowed
from mariner.schemas.token import Token
from mariner.schemas.user_schemas import (
    User,
    UserCreateBasic,
    UserCreateOAuth,
    UserUpdate,
)
from mariner.stores.oauth_state_sql import oauth_state_store
from mariner.stores.user_sql import user_store


def get_users(
    db: Session,
    skip: int = 0,
    limit: int = 100,
):
    return user_store.get_multi(db, skip=skip, limit=limit)


def create_user_basic(db: Session, request: UserCreateBasic):
    user = user_store.create(db, obj_in=request)
    return User.from_orm(user)


def update_user(db: Session, request: UserUpdate, user: UserEntity) -> User:
    current_user_data = jsonable_encoder(user)
    user_in = UserUpdate(**request.dict(), **current_user_data)
    user = user_store.update(db, db_obj=user, obj_in=user_in)
    return User.from_orm(user)


def get_user(db: Session, user_id: int, current_user: UserEntity) -> User:
    if not user_store.is_superuser(current_user):
        raise UserNotSuperUser()
    user = user_store.get(db, id=user_id)
    if not user:
        raise UserNotFound()
    return User.from_orm(user)


class BasicAuth(BaseModel):
    username: str
    password: str


class GithubAuth(BaseModel):
    code: str
    state: str


def _authenticate_basic(email: str, password: str) -> User:
    with SessionLocal() as db:
        user = user_store.authenticate(db, email=email, password=password)
        if not user:
            raise UserNotFound()
        if not user_store.is_active(user):
            raise UserNotActive()
        return User.from_orm(user)


def _authenticate_github(code: str, state: str) -> User:
    with SessionLocal() as db:
        state = oauth_state_store.get_state(db, state=state, provider="github")
        if not state:
            raise InvalidOAuthState(f"state {state} not created by us")

        token = github.get_access_token(code)
        github_user = github.get_user(token.access_token)
        if github_user.email not in settings.ALLOWED_GITHUB_AUTH_EMAILS:
            raise UserEmailNotAllowed(
                "Email should be cleared in the ALLOWED_GITHUB_AUTH_EMAILS env var"
            )

        user = user_store.get_by_email(db, email=github_user.email)
        if not user:
            user = user_store.create(
                db,
                obj_in=UserCreateOAuth(
                    email=EmailStr(github_user.email), image_url=github_user.avatar_url
                ),
            )
        if not user.is_active:
            raise UserNotActive()
        return User.from_orm(user)


def _make_token(user: User) -> Token:
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return Token(
        access_token=security.create_access_token(
            user.id, expires_delta=access_token_expires
        ),
        token_type="bearer",
    )


def authenticate(
    basic: Optional[BasicAuth] = None, github_oauth: Optional[GithubAuth] = None
) -> Token:
    if github_oauth:
        user = _authenticate_github(code=github_oauth.code, state=github_oauth.state)
    elif basic:
        user = _authenticate_basic(email=basic.username, password=basic.password)
    else:
        raise ValueError("authenticate must be called with some strategy")
    return _make_token(user)
