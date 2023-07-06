"""
Exposes the oauth functions abstracting the provider.
"""
from typing import Union

from pydantic import BaseModel

from oauth_providers import github


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
        **kwargs: the arguments to pass to the provider.
    """
    if provider == "github":
        github_user = github.exchange_code(**kwargs)
        return UserData(email=github_user.email, avatar_url=github_user.avatar_url)
    raise NotImplementedError(f"provider {provider} not implemented")
