from sqlalchemy.orm import Session

from mariner.core.config import settings
from mariner.entities.user import User
from mariner.schemas.user_schemas import UserCreateBasic
from mariner.stores.user_sql import user_store


def get_test_user(db: Session) -> User:
    user = user_store.get_by_email(db, email=settings.EMAIL_TEST_USER)
    if not user:
        user = user_store.create(
            db,
            obj_in=UserCreateBasic(email=settings.EMAIL_TEST_USER, password="123456"),
        )
    assert user is not None
    return user
