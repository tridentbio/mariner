"""
User data layer defining ways to read and write to the users collection
"""

from typing import Any, Dict, Optional, Union

from sqlalchemy.orm import Session

from mariner.core.security import get_password_hash, verify_password
from mariner.entities.user import User
from mariner.schemas.user_schemas import (
    UserCreateBasic,
    UserCreateOAuth,
    UserUpdate,
)
from mariner.stores.base_sql import CRUDBase


class _CRUDUser(CRUDBase[User, UserCreateBasic, UserUpdate]):
    def get_by_email(self, db: Session, *, email: str) -> Optional[User]:
        """Gets a single user by email.

        Args:
            db: Connection to the datamase.
            email: Email to use in search.

        Returns:
            User with input email.
        """
        return db.query(User).filter(User.email == email).first()

    def create(
        self, db: Session, *, obj_in: Union[UserCreateBasic, UserCreateOAuth]
    ) -> User:
        """Persists user in database.

        Args:
            db: Connection to the database.
            obj_in: User creation object>

        Returns:
            Created user.
        """
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
        """Updates user

        Args:
            db: Connection to the database
            db_obj: User instance to be updated
            obj_in: Update object

        Returns:
            Updated user
        """
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
        """Gets the user linked to email if the password is correct.

        Args:
            db: Connection to the database.
            email: Email of the user.
            password: Password of the user

        Returns:
            User instance if authentication succeeds.
        """
        user = self.get_by_email(db, email=email)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    def is_active(self, user: User) -> bool:
        """Checks if the user is active.

        Args:
            user: User to be checked.

        Returns:
            True if user is active. False otherwise.
        """
        return user.is_active

    def is_superuser(self, user: User) -> bool:
        """Checks if a user is super user.

        Args:
            user: User to be checked

        Returns:
            True if the user is super user, False otherwise.
        """
        return user.is_superuser


user_store = _CRUDUser(User)
