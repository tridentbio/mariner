"""
Useful functions when developing on a fresh database
"""
import sqlalchemy.exc
from sqlalchemy.orm import Session

from mariner.core.config import settings
from mariner.core.security import get_password_hash
from mariner.db.session import SessionLocal
from mariner.entities.user import User


def init_db(db: Session) -> None:
    """
    Function is deprecated to be used in production as long
    as using alembic migrations
    """
    # Tables should be created with Alembic migrations
    # But if you don't want to use migrations, create
    # the tables un-commenting the next line
    # Base.metadata.create_all(bind=engine)


def create_user(email: str, password: str, superuser: bool = False):
    user = User(
        email=email,
        hashed_password=get_password_hash(password),
        is_active=True,
        is_superuser=superuser,
    )
    with SessionLocal() as db:
        try:
            db.add(user)
            db.flush()
            db.refresh(user)
            db.commit()
        except sqlalchemy.exc.IntegrityError:
            print(f"user {email} already exists")

    return user


def create_admin_user() -> User:
    """Creates an admin user to be used for local development

    This functions is not called from the app itself, but by developers.
    An easy way to call this function from command line:

        ``python -c 'from mariner.db.init_db impor create_admin_user; create_admin_user()'``

    User created has following credentials

    - Email: admin@mariner.trident.bio
    - Password: 123456

    Returns:
        the super user entity
    """
    return create_user(
        email="admin@mariner.trident.bio", password="123456", superuser=True
    )


def create_test_user():
    """Creates a test user to be used for local development

    This functions is not called from the app itself, but by developers
    during developers.
    An easy way to call this function from command line:

        python -c 'from mariner.db.init_db impor create_test_user; create_test_user()'

    User created has following credentials
    Email: :attribute:`settings.EMAIL_TEST_USER`
    Password: 123456

    Returns:
        the super user entity
    """
    return create_user(email=settings.EMAIL_TEST_USER, password="123456")
