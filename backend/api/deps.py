"""
Utility FastAPI dependencies

Useful for intercept requests before controller handlers, .e.g. performing
authentication
"""
from typing import Union

from fastapi import Header, WebSocket, status
from fastapi.exceptions import HTTPException
from fastapi.param_functions import Depends, Query
from fastapi.security.oauth2 import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import ValidationError
from sqlalchemy.orm import Session

from mariner.core import security
from mariner.core.config import settings
from mariner.db.session import SessionLocal
from mariner.entities.deployment import Deployment
from mariner.entities.user import User
from mariner.schemas.token import TokenPayload
from mariner.stores.deployment_sql import deployment_store
from mariner.stores.user_sql import user_store

reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/login/access-token"
)


def get_db():
    """DB generator with auto-closing."""
    db = SessionLocal()
    yield db
    db.close()


def get_current_user(
    db: Session = Depends(get_db), token: str = Depends(reusable_oauth2)
) -> User:
    """Get's the user from a authentication token.

    Args:
        db: Connection to the database.
        token: String with authentication.

    Returns:
        The same input user if he is authenticated.

    Raises:
        HTTPException: 400 when user is not active.
    """
    try:
        payload = jwt.decode(
            token, settings.AUTHENTICATION_SECRET_KEY, algorithms=[security.ALGORITHM]
        )
        token_data = TokenPayload(**payload)
    except (JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    user = user_store.get(db, id=token_data.sub)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Authenticates if the user is active or raise HTTPException

    Args:
        current_user: User that originated request.

    Returns:
        The same input user if he is authenticated.

    Raises:
        HTTPException: 400 when user is not active.
    """
    if not user_store.is_active(current_user):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def get_current_active_superuser(
    current_user: User = Depends(get_current_user),
) -> User:
    """Authenticates if the user is superuser or raise HTTPException

    Args:
        current_user: User that originated request.

    Returns:
        The same input user if he is authenticated.

    Raises:
        HTTPException: 400 when user is not super user.
    """
    if not user_store.is_superuser(current_user):
        raise HTTPException(
            status_code=400, detail="The user doesn't have enough privileges"
        )
    return current_user


async def get_cookie_or_token(
    websocket: WebSocket,
    token: Union[str, None] = Query(default=None),
):
    """Gets the token from a http websocket connection request.

    Args:
        websocket: instance to send and receive messages.
        token: the token present in the message received.
    """
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
    return token


def assert_trusted_service(authorization: Union[str, None] = Header("Authorization")):
    """Checks a basic inter service authentication using the Authorization HTTP header.

    Args:
        authorization: string parsed from request headers.

    Raises:
        HTTPException: 403 when the token doesn't match
    """
    if not authorization:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    else:
        token = authorization.split(" ")
        if len(token) < 2 or token[1] != settings.APPLICATION_SECRET:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)


def get_current_websocket_user(
    db: Session = Depends(get_db), token: str = Depends(get_cookie_or_token)
) -> User:
    """Gets the user entity encoded in an authentication token.

    Args:
        db: Connection to the database.
        token: Authentication token passed in request.

    Returns:
        User from token.
    """
    return get_current_user(db, token)


def get_current_websocket_deployment(
    db: Session = Depends(get_db), token: str = Depends(get_cookie_or_token)
) -> Deployment:
    """Gets the deployment entity encoded in an authentication token.

    Args:
        db: Connection to the database.
        token: Authentication token passed in request.

    Returns:
        Deployment from token.

    Raises:
        HTTPException: 403 authentication error.
    """
    try:
        assert token is not None
        payload = security.decode_deployment_url_token(token)

        deployment = deployment_store.get(db, id=payload.sub)
        assert deployment.share_strategy == "public"

        return deployment
    except (AssertionError, JWTError, ValidationError) as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN) from e
