"""
Users service
"""

from datetime import timedelta
from typing import Optional

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

import oauth_providers
from mariner.core import security
from mariner.core.config import get_app_settings
from mariner.db.session import SessionLocal
from mariner.entities.user import User as UserEntity
from mariner.exceptions import (
    InvalidOAuthState,
    UserNotActive,
    UserNotFound,
    UserNotSuperUser,
)
from mariner.exceptions.user_exceptions import UserEmailNotAllowed
from mariner.oauth import oauth_manager
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
    """Gets a paginated list of users.

    Only usable to super users.

    .. deprecated::0.1.0


    Args:
        db: Connection to the database.
        skip: Number of items to skip.
        limit: Number of items in the result.
    """
    return user_store.get_multi(db, skip=skip, limit=limit)


def create_user_basic(db: Session, request: UserCreateBasic):
    """Creates user through basic auth, i.e. email and password.

    Args:
        db: Connection to the database.
        request: Request with user data.
    """
    user = user_store.create(db, obj_in=request)
    return User.from_orm(user)


def update_user(db: Session, request: UserUpdate, user: UserEntity) -> User:
    """Updates user with request information.

    Args:
        db: Connection to the database.
        request: Request with update data.
        user: User that originated the request.

    Returns:
        Updated user.
    """
    current_user_data = jsonable_encoder(user)
    user_in = UserUpdate(**request.dict(), **current_user_data)
    user = user_store.update(db, db_obj=user, obj_in=user_in)
    return User.from_orm(user)


def get_user(db: Session, user_id: int, current_user: UserEntity) -> User:
    """Gets a single user.

    Args:
        db: Connection to the database.
        user_id: Id of the user to get.
        current_user: User that originated the request.

    Returns:
        User with user_id as id.

    Raises:
        UserNotSuperUser: If user is not trying to get own data, and user is not superuser.
        UserNotFound: If there's no user with id equals user_id.
    """
    if not user_store.is_superuser(current_user):
        raise UserNotSuperUser()
    user = user_store.get(db, id=user_id)
    if not user:
        raise UserNotFound()
    return User.from_orm(user)


class BasicAuth(BaseModel):
    """Payload for basic authentication request."""

    username: str
    password: str


def _authenticate_basic(email: str, password: str) -> User:
    with SessionLocal() as db:
        user = user_store.authenticate(db, email=email, password=password)
        if not user:
            raise UserNotFound()
        if not user_store.is_active(user):
            raise UserNotActive()
        return User.from_orm(user)


def _authenticate_oauth(state: str, **kwargs) -> User:
    with SessionLocal() as db:
        state_obj = oauth_state_store.get_state(db, state=state)
        if not state:
            raise InvalidOAuthState(f"state {state} not created by us")
        provider_settings = get_app_settings("auth").__root__[
            state_obj.provider
        ]
        user_data = oauth_providers.get_user_data(
            provider=state_obj.provider,
            credentials=(
                provider_settings.dict()
                | {"redirect_uri": oauth_manager.redirect_uri}
            ),
            **kwargs,
        )
        if (
            provider_settings.allowed_emails
            and user_data.email not in provider_settings.allowed_emails
        ):
            raise UserEmailNotAllowed(
                f"Email missing from conf.toml's {state_obj.provider} settings"
            )

        user = user_store.get_by_email(db, email=user_data.email)
        if not user:
            user = user_store.create(
                db,
                obj_in=UserCreateOAuth(
                    email=EmailStr(user_data.email),
                    image_url=user_data.avatar_url,
                ),
            )
        if not user.is_active:
            raise UserNotActive()
        return User.from_orm(user)


def _make_token(user: User) -> Token:
    access_token_expires = timedelta(
        minutes=get_app_settings("server").access_token_expire_minutes
    )
    return Token(
        access_token=security.create_access_token(
            user.id, expires_delta=access_token_expires
        ),
        token_type="bearer",
    )


def authenticate(
    basic: Optional[BasicAuth] = None, provider_oauth: Optional[dict] = None
) -> Token:
    """Authenticates one of the input authentication payloads (basic, or provider_oauth)

    Args:
        basic: If present, attempts a basic authentication.
        provider_oauth: If present, attempts a provider authentication. **Precedes basic**.

    Returns:
        Object with access token.

    Raises:
        ValueError: When all authentication input options are None.
    """
    if provider_oauth:
        if "state" not in provider_oauth:
            raise ValueError("Missing state in provider_oauth")
        user = _authenticate_oauth(**provider_oauth)
    elif basic:
        user = _authenticate_basic(
            email=basic.username, password=basic.password
        )
    else:
        raise ValueError("authenticate must be called with some strategy")
    return _make_token(user)
