from typing import Union

from fastapi import Header, WebSocket, status
from fastapi.exceptions import HTTPException
from fastapi.param_functions import Depends, Query
from fastapi.security.oauth2 import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import ValidationError
from sqlalchemy.orm import Session

from mariner import schemas
from mariner.core import security
from api.config import settings
from mariner.db.session import SessionLocal
from mariner.stores.user_sql import user_store
from mariner.entities.user import User

reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/login/access-token"
)


def get_db():
    db = SessionLocal()
    yield db
    db.close()


def get_current_user(
    db: Session = Depends(get_db), token: str = Depends(reusable_oauth2)
) -> User:
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[security.ALGORITHM]
        )
        token_data = schemas.TokenPayload(**payload)
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
    if not user_store.is_active(current_user):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def get_current_active_superuser(
    current_user: User = Depends(get_current_user),
) -> User:
    if not user_store.is_superuser(current_user):
        raise HTTPException(
            status_code=400, detail="The user doesn't have enough privileges"
        )
    return current_user


async def get_cookie_or_token(
    websocket: WebSocket,
    token: Union[str, None] = Query(default=None),
):
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
    return token


def assert_trusted_service(authorization: Union[str, None] = Header("Authorization")):
    if not authorization:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    else:
        token = authorization.split(" ")
        if len(token) < 2 or token[1] != settings.APPLICATION_SECRET:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)


def get_current_websocket_user(
    db: Session = Depends(get_db), token: str = Depends(get_cookie_or_token)
) -> User:
    return get_current_user(db, token)
