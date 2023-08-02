from sqlalchemy.orm import Session

from mariner.core.config import get_app_settings
from mariner.core.security import get_password_hash
from mariner.entities.user import User
from mariner.schemas.user_schemas import UserCreateBasic
from mariner.stores.user_sql import user_store
from tests.utils.utils import random_lower_string


def get_test_user(db: Session) -> User:
    user = user_store.get_by_email(db, email=get_app_settings("test").email_test_user)
    if not user:
        user = user_store.create(
            db,
            obj_in=UserCreateBasic(
                email=get_app_settings("test").email_test_user, password="123456"
            ),
        )
    assert user is not None
    return user


def get_random_test_user(db: Session) -> User:
    hashed_password = get_password_hash("123456")
    user = User(
        email=f"{random_lower_string()}@domain.com",
        is_active=True,
        is_superuser=True,
        hashed_password=hashed_password,
    )
    db.add(user)
    db.flush()
    db.refresh(user)
    db.commit()
    return user
