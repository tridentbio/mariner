"""
Auxiliary functions related to OAuth 2.0 authentication.
"""
from collections.abc import Mapping
from typing import Dict, Union

from sqlalchemy.orm import Session

from mariner.core.config import AuthSettings, get_app_settings
from mariner.db.session import SessionLocal
from mariner.stores.oauth_state_sql import oauth_state_store


class OAuthManager(Mapping):
    """
    Stores all OAuth providers, their urls and secrets.
    """

    auth_providers: Dict[str, AuthSettings]

    def __init__(self, auth_providers: Union[None, Dict[str, AuthSettings]] = None):
        if not auth_providers:
            self.auth_providers = get_app_settings("auth")
        else:
            self.auth_providers = auth_providers

    def __getitem__(self, key):
        return self.auth_providers[key]

    def __len__(self):
        return len(self.auth_providers)

    def __contains__(self, key):
        return key in self.auth_providers

    def __iter__(self):
        return iter(self.auth_providers)

    def get_redirect_uri(self, key: str):
        """
        Builds a oauth url with attributes from oauth_settings and a state.

        Args:
            oauth_settings: A dictionary with the attributes to build the url.
        """
        with SessionLocal() as db:

            state = oauth_state_store.create_state(db, provider=key).state
            redirect_uri = f"{get_app_settings('server').host}/api/v1/oauth-callback"
            oauth_settings = self[key]
            return (
                f"{oauth_settings.authorization_url}?client_id={oauth_settings.client_id}"
                f"&redirect_uri={redirect_uri}&scope={oauth_settings.scope}&state={state}"
            )


oauth_manager = OAuthManager()
