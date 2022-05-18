from sqlalchemy.orm import Session

from app.core.config import settings
from app.features.user.exceptions import UserAlreadyExists
from app.utils import send_new_account_email

from .crud import repo
from .schema import UserCreate


def get_users(
    db: Session,
    skip: int = 0,
    limit: int = 100,
):
    print(skip, limit)
    return repo.get_multi(db, skip=skip, limit=limit)


def create_user(
    db: Session,
    user_in: UserCreate,
):
    user = repo.get_by_email(db, email=user_in.email)
    if user:
        raise UserAlreadyExists
    print("gonna create now")
    user = repo.create(db, obj_in=user_in)
    print("created")
    if settings.EMAILS_ENABLED and user_in.email:
        send_new_account_email(
            email_to=user_in.email, username=user_in.email, password=user_in.password
        )
    return user
