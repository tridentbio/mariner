from typing import Any, Dict, Optional, Union

from sqlalchemy.orm import Session, class_mapper

from mariner.core.security import get_password_hash, verify_password
from mariner.entities.user import User
from mariner.schemas.user_schemas import UserCreateBasic, UserCreateOAuth, UserUpdate
from mariner.stores.base_sql import CRUDBase


class _CRUDUser(CRUDBase[User, UserCreateBasic, UserUpdate]):
    def get_by_email(self, db: Session, *, email: str) -> Optional[User]:
        return db.query(User).filter(User.email == email).first()

    def create(
        self, db: Session, *, obj_in: Union[UserCreateBasic, UserCreateOAuth]
    ) -> User:
        db_obj = User(
            email=obj_in.email,
            hashed_password=(
                get_password_hash(obj_in.password)
                if isinstance(obj_in, UserCreateBasic)
                else None
            ),
            full_name=obj_in.full_name,
            is_superuser=obj_in.is_superuser,
        )
        db.add(db_obj)
        db.flush()
        db.refresh(db_obj)
        db.commit()
        return db_obj

    def update(
        self, db: Session, *, db_obj: User, obj_in: Union[UserUpdate, Dict[str, Any]]
    ) -> User:
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)
        if "password" in update_data:
            hashed_password = get_password_hash(update_data["password"])
            del update_data["password"]
            update_data["hashed_password"] = hashed_password
        return super().update(db, db_obj=db_obj, obj_in=update_data)

    def authenticate(self, db: Session, *, email: str, password: str) -> Optional[User]:
        user = self.get_by_email(db, email=email)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    def is_active(self, user: User) -> bool:
        return user.is_active

    def is_superuser(self, user: User) -> bool:
        return user.is_superuser


user_store = _CRUDUser(User)
