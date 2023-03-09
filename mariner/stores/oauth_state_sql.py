"""
OAuth state data layer defining ways to read and write to the state collection.
"""

import os
from typing import Any, Optional

from sqlalchemy import and_
from sqlalchemy.orm import Session

from mariner.entities import OAuthState
from mariner.stores.base_sql import CRUDBase


class CRUDOAuthState(CRUDBase[OAuthState, Any, Any]):
    """Data layer operations on the collection of oauth state keys."""

    def create_state(self, db: Session, provider: str):
        """Creates a secret string state for OAuth purposes.

        Function to create a secret random string persisted to the database
        to check if OAuth attempts that were issued through this own server redirect
        call.



        Args:
            db: Connection to database.
            provider: String identifying the provider taking part in oauth.

        Return
        """
        state = OAuthState(state=os.urandom(16).hex(), provider=provider)

        db.add(state)
        db.flush()
        db.commit()
        return state

    def get_state(self, db: Session, state: str, provider: Optional[str] = None):
        """Gets a single instance of state entity that matches the state string.

        Args:
            db: Connection to the database.
            state: State string generated previously by this server.
            provider: String identifier of the provider. If none is passed, search over all
            states.
        """
        query = OAuthState.state == state
        if provider:
            query = and_(query, OAuthState.provider == provider)
        return db.query(OAuthState).filter(query).first()


oauth_state_store = CRUDOAuthState(OAuthState)
