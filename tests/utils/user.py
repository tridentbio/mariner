from typing import Dict

from fastapi.testclient import TestClient
from pydantic.networks import EmailStr
from sqlalchemy.orm import Session

from mariner.core.config import settings
from mariner.entities import User
from mariner.schemas.user_schemas import UserCreate, UserUpdate
from mariner.stores.user_sql import user_store
from tests.utils.utils import random_email, random_lower_string


def user_authentication_headers(
    *, client: TestClient, email: str, password: str
) -> Dict[str, str]:
    data = {"username": email, "password": password}

    r = client.post(f"{settings.API_V1_STR}/login/access-token", data=data)
    response = r.json()
    auth_token = response["access_token"]
    headers = {"Authorization": f"Bearer {auth_token}"}
    return headers


def create_random_user(db: Session) -> User:
    email = random_email()
    password = random_lower_string()
    user_in = UserCreate(email=EmailStr(email), password=password)
    user = user_store.create(db=db, obj_in=user_in)
    return user


def authentication_token_from_email(
    *, client: TestClient, email: str, db: Session
) -> Dict[str, str]:
    """
    Return a valid token for the user with given email.

    If the user doesn't exist it is created first.
    """
    password = random_lower_string()
    user = user_store.get_by_email(db, email=email)
    if not user:
        user_in_create = UserCreate(email=EmailStr(email), password=password)
        user = user_store.create(db, obj_in=user_in_create)
    else:
        user_in_update = UserUpdate(password=password)
        user = user_store.update(db, db_obj=user, obj_in=user_in_update)

    return user_authentication_headers(client=client, email=email, password=password)
