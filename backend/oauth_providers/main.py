"""
Exposes the oauth functions abstracting the provider.
"""
from typing import Union

from pydantic import BaseModel

from oauth_providers import genentech, github


class UserData(BaseModel):
    """
    Data of a user retrieved from a provider."""

    email: str
    avatar_url: Union[None, str] = None


def get_user_data(provider: str, **kwargs) -> UserData:
    """
    Returns the user data from a provider.

    Args:
        provider (str): the provider to use.
        code (str): the code to exchange for a user. (keyword!)
        credentials (dict[str, str]): the credentials to use from conf.toml. (keyword!)
        **kwargs: the arguments to pass to the provider. It must include.
    """
    if provider == "github":
        github_user = github.exchange_code(**kwargs)
        return UserData(email=github_user.email, avatar_url=github_user.avatar_url)
    elif provider == "genentech":
        client = genentech.GenentechClient(
            genentech.ClientOptions(
                client_id=kwargs["client_id"],
                client_secret=kwargs["client_secret"],
                redirect_uri=kwargs["redirect_uri"],
                jwks_url=kwargs["jwks_url"],
                token_url=kwargs["token_url"],
            )
        )
        genentech_user = client.exchange_code(kwargs["code"])
        return UserData(email=genentech_user.email)
    raise NotImplementedError(f"provider {provider} not implemented")
