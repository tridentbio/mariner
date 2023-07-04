"""
Auxiliary functions related to OAuth 2.0 authentication
"""
from typing import MutableMapping

from sqlalchemy.orm import Session

from mariner.core.config import get_app_settings
from mariner.stores.oauth_state_sql import oauth_state_store


def get_github_oauth_url(db: Session):
    """Gets github OAuth url

    Needs database connection to write a secret state giving clearence to
    next github request.

    Args:
        db: Connection with database
    """
    state = oauth_state_store.create_state(db, provider="github").state
    github_secrets = get_app_settings("auth")
    redirect_uri = f"{get_app_settings('server').host}/api/v1/oauth-callback"
    return f"https://github.com/login/oauth/authorize?client_id={github_secrets.client_id}&redirect_uri={redirect_uri}&scope=read:user,user:email&state={state}"  # noqa E501


class OAuthProviders(MutableMapping):
    def __init__(self):
        self.provider_url_makers = {"github": get_github_oauth_url}

    def __getitem__(self, key):
        return self.provider_url_makers[key]

    def __setitem__(self, key, value):
        self.provider_url_makers[key] = value

    def __len__(self):
        return len(self.provider_url_makers)

    def __delitem__(self, key):
        del self.provider_url_makers[key]

    def __iter__(self):
        return iter(self.provider_url_makers)


provider_url_makers = OAuthProviders()
