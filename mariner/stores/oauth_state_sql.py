import os
from typing import Any, Optional
from sqlalchemy import and_
from sqlalchemy.orm import Session
from sqlalchemy.orm.mapper import class_mapper
from mariner.entities import OAuthState
from mariner.stores.base_sql import CRUDBase


class CRUDOAuthState(CRUDBase[OAuthState, Any, Any]):
    def create_state(self, db: Session, provider: str):
        state = OAuthState(state=os.urandom(16).hex(), provider=provider)

        db.add(state)
        db.flush()
        db.commit()
        return state

    def get_state(self, db: Session, state: str, provider: Optional[str] = None):
        query = OAuthState.state == state
        if provider:
            query = and_(query, OAuthState.provider == provider)
        return db.query(OAuthState).filter(query).first()


oauth_state_store = CRUDOAuthState(OAuthState)
