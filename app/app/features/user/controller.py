from fastapi.encoders import jsonable_encoder
from fastapi.param_functions import Depends
from sqlalchemy.orm import Session
from app.core.config import settings

from app.features.user.exceptions import UserAlreadyExists
from app.utils import send_new_account_email

from .crud import repo 
from .schema import UserCreate
from .model import User

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
    print('gonna create now')
    user = repo.create(db, obj_in=user_in)
    print('created')
    if settings.EMAILS_ENABLED and user_in.email:
        send_new_account_email(
                email_to=user_in.email, username=user_in.email, password=user_in.password)
    return user

def update_user(db: Session, current_user: User):
    current_user_data = jsonable_encoder(current_user)
    user_in = schemas.UserUpdate(**current_user_data)
    if password is not None:
        user_in.password = password
    if full_name is not None:
        user_in.full_name = full_name
    if email is not None:
        user_in.email = email
    user = repo.update(db, db_obj=current_user, obj_in=user_in)

