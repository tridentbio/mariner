"""
Useful functions when developing on a fresh database
"""
from sqlalchemy.orm import Session

from mariner.core.security import get_password_hash
from mariner.db.session import SessionLocal
from mariner.entities.user import User


def init_db(db: Session) -> None:
    # Tables should be created with Alembic migrations
    # But if you don't want to use migrations, create
    # the tables un-commenting the next line
    # Base.metadata.create_all(bind=engine)
    pass


def create_admin_user() -> User:
    with SessionLocal() as db:
        hashed_password = get_password_hash("123456")
        user = User(
            email="admin@mariner.trident.bio",
            is_active=True,
            is_superuser=True,
            hashed_password=hashed_password,
        )
        db.add(user)
        db.flush()
        db.refresh(user)
        db.commit()
        return user
