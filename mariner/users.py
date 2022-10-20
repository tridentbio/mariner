from sqlalchemy.orm import Session
from mariner.exceptions import UserAlreadyExists

from mariner.stores.user_sql import UserCreate, user_store


def get_users(
    db: Session,
    skip: int = 0,
    limit: int = 100,
):
    return user_store.user_store.get_multi(db, skip=skip, limit=limit)


def create_user(
    db: Session,
    user_in: UserCreate,
):
    user = user_store.get_by_email(db, email=user_in.email)
    if user:
        raise UserAlreadyExists
    user = user_store.create(db, obj_in=user_in)
    if settings.EMAILS_ENABLED and user_in.email:
        send_new_account_email(
            email_to=user_in.email, username=user_in.email, password=user_in.password
        )
    return user
