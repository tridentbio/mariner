"""
Auxiliary functions related to OAuth 2.0 authentication.
"""
import urllib.parse
from collections.abc import Mapping
from typing import Dict, List, Union

from pydantic import BaseModel

from mariner.core.config import (
    AuthSettings,
    AuthSettingsDict,
    get_app_settings,
)
from mariner.db.session import SessionLocal
from mariner.stores.oauth_state_sql import oauth_state_store


class Provider(BaseModel):
    """
    Provider's information needed by the frontend.

    Attributes:
        id: The provider's id. Should match one of the keys in the
            auth_providers dict.
        logo_url: The url to get the provider's logo.
    """

    id: str
    logo_url: Union[None, str]
    name: str


class OAuthManager(Mapping):
    """
    Stores all OAuth providers, their urls and secrets.
    """

    auth_providers: Dict[str, AuthSettings]

    def __init__(
        self, auth_providers: Union[None, Dict[str, AuthSettings]] = None
    ):
        self.redirect_uri = (
            f"{get_app_settings('server').host}/api/v1/oauth-callback"
        )
        if not auth_providers:
            self.auth_providers = get_app_settings("auth").__root__
        else:
            self.auth_providers = AuthSettingsDict.parse_obj(
                auth_providers
            ).__root__

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
            oauth_settings = self[key]
            params = {
                "client_id": oauth_settings.client_id,
                "redirect_uri": self.redirect_uri,
                "state": state,
                "response_type": "code",
                "scope": oauth_settings.scope,
            }
            querysting = urllib.parse.urlencode(params)

            return f"{oauth_settings.authorization_url}?{querysting}"

    def get_providers(self) -> List[Provider]:
        """
        Returns a list of provider's non-sensitive information.
        """
        providers = []
        for provider_id, provider in self.auth_providers.items():
            provider = self[provider_id]
            providers.append(
                Provider(
                    id=provider_id,
                    logo_url=provider.logo_url,
                    name=provider.name,
                )
            )
        return providers


oauth_manager = OAuthManager()
