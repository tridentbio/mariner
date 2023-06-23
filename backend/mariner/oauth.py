"""
Auxiliary functions related to OAuth 2.0 authentication
"""
from sqlalchemy.orm import Session

from mariner.core.config import settings
from mariner.stores.oauth_state_sql import oauth_state_store


def get_github_oauth_url(db: Session):
    """Gets github OAuth url

    Needs database connection to write a secret state giving clearence to
    next github request.

    Args:
        db: Connection with database
    """
    state = oauth_state_store.create_state(db, provider="github").state
    redirect_uri = f"{settings.SERVER_HOST}/api/v1/oauth-callback"
    return f"https://github.com/login/oauth/authorize?client_id={settings.GITHUB_CLIENT_ID}&redirect_uri={redirect_uri}&scope=read:user,user:email&state={state}"  # noqa E501


# Dictionary with URL building functions
provider_url_makers = {"github": get_github_oauth_url}
