from sqlalchemy.orm import Session

from mariner.stores.user_sql import user_store


def get_users(
    db: Session,
    skip: int = 0,
    limit: int = 100,
):
    return user_store.get_multi(db, skip=skip, limit=limit)
