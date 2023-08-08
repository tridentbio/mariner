"""
Github service
"""
from logging import error

import requests
from pydantic import BaseModel

GITHUB_URL = "https://github.com/"
GITHUB_API_URL = "https://api.github.com/"


class InvalidGithubCode(Exception):
    """Exception raised when the github token from the request is invalid
    i.e. it didn't came from github."""


class GithubUserUnverified(Exception):
    """Exception raised when the github user is not verified."""


def _make_headers(**kwargs):
    headers = {"Accept": "application/json"}
    if "access_token" in kwargs:
        access_token = kwargs["access_token"]
        headers["Authorization"] = f"Bearer {access_token}"
        del kwargs["access_token"]
    return headers, kwargs


def _join(url: str, path: str):
    if not url.endswith("/"):
        url += "/"
    if path.startswith("/"):
        path.removesuffix("/")
    return url + path


def github_get(
    path: str, client_id: str, client_secret: str, url=GITHUB_URL, **kwargs
) -> requests.Response:
    """Makes GET/ request to github on path endpoint.

    Args:
        path: Endpoint used in request.
        url (str): Base url.

    Returns:
        The request response.
    """
    headers, kwargs = _make_headers(**kwargs)
    kwargs["client_id"] = client_id
    kwargs["client_secret"] = client_secret
    url = _join(url, path)
    return requests.get(url=url, params=kwargs, headers=headers, timeout=10)


def github_post(
    path: str, client_id: str, client_secret: str, url=GITHUB_URL, **kwargs
) -> requests.Response:
    """Makes a POST/ request to github on path endpoint.

    Args:
        path: Endpoint used in request
        url (str): Base url.

    Returns:
        The request response.
    """
    headers, kwargs = _make_headers(**kwargs)
    kwargs["client_id"] = client_id
    kwargs["client_secret"] = client_secret

    result = requests.post(
        url=url + path, json=kwargs, headers=headers, timeout=5
    )
    return result


class GithubAccessCode(BaseModel):
    """Model of an access code object from github, used to make requests
    in behalf of a user.
    """

    access_token: str
    scope: str
    token_type: str


# Complete payload:
# https://docs.github.com/en/rest/users/users#get-the-authenticated-user
class GithubUser(BaseModel):
    """Models a github user."""

    id: int
    login: str
    email: str
    bio: str
    name: str
    gravatar_id: str
    avatar_url: str


class GithubFailure(Exception):
    """Exception raised when github returns an error."""

    def __init__(self, message: str, response: requests.Response):
        super().__init__()
        self.message = message
        self.response = response


def get_access_token(
    code: str, credentials: dict[str, str]
) -> GithubAccessCode:
    """Attempts to exchange a code string for an access token.

    Args:
        code: code string used during oauth.

    Returns:
        GithubAccessCode object with github credentials.

    Raises:
        InvalidGithubCode: If the given input is invalid.
    """
    result = github_post("/login/oauth/access_token", code=code, **credentials)
    if 200 <= result.status_code < 400:
        return GithubAccessCode.construct(**result.json())
    raise InvalidGithubCode()


def get_user(
    access_token: str,
    credentials: dict[str, str],
) -> GithubUser:
    """Get's the user github owner of the access_token.

    Args:
        access_token: github authentication token.
        credentials: a dictionary containing the client_id and the
            client_secret.

    Returns:
        Github user object

    Raises:
        GithubFailure: When there's an error from github response.
    """
    result = github_get(
        url=GITHUB_API_URL,
        path="user",
        access_token=access_token,
        **credentials,
    )
    if 200 <= result.status_code < 400:
        github_user = GithubUser.construct(**result.json())
        if not github_user.email:
            email_response = github_get(
                url=GITHUB_API_URL,
                path="user/emails",
                access_token=access_token,
                **credentials,
            )
            assert (
                200 <= email_response.status_code < 400
            ), "Get user's email from github failed'"
            email_json = email_response.json()
            primary_email = None
            for email in email_json:
                if email["verified"] and email["primary"]:
                    primary_email = email["email"]
                    break
            if not primary_email:
                raise GithubUserUnverified(
                    "No primary email verified found for user"
                )
            github_user.email = primary_email
        return github_user
    else:
        error(result.json())
        raise GithubFailure("Failed to get result", response=result)


def exchange_code(code: str, credentials: dict[str, str]) -> GithubUser:
    """Exchanges a code for a github user.

    Args:
        code: code string used during oauth.
        credentials: a dictionary containing the client_id and the
            client_secret.

    Returns:
        Github user object
    """
    access_token = get_access_token(code, credentials).access_token
    return get_user(access_token, credentials)
