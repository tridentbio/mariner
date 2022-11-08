from logging import error

import requests
from pydantic import BaseModel

from mariner.core.config import settings
from mariner.exceptions.auth_exceptions import InvalidGithubCode

GITHUB_URL = "https://github.com/"
GITHUB_API_URL = "https://api.github.com/"


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


def github_get(path: str, url=GITHUB_URL, **kwargs) -> requests.Response:
    headers, kwargs = _make_headers(**kwargs)
    kwargs["client_id"] = settings.GITHUB_CLIENT_ID
    kwargs["client_secret"] = settings.GITHUB_CLIENT_SECRET
    url = _join(url, path)
    return requests.get(url=url, params=kwargs, headers=headers)


def github_post(path: str, url=GITHUB_URL, **kwargs) -> requests.Response:
    headers, kwargs = _make_headers(**kwargs)
    kwargs["client_id"] = settings.GITHUB_CLIENT_ID
    kwargs["client_secret"] = settings.GITHUB_CLIENT_SECRET

    result = requests.post(url=url + path, json=kwargs, headers=headers)
    return result


class GithubAccessCode(BaseModel):
    access_token: str
    scope: str
    token_type: str


# Complete payload: https://docs.github.com/en/rest/users/users#get-the-authenticated-user
class GithubUser(BaseModel):
    id: int
    login: str
    email: str
    bio: str
    name: str
    gravatar_id: str
    avatar_url: str


class GithubFailure(Exception):
    def __init__(self, message: str, response: requests.Response):
        super().__init__()
        self.message = message
        self.response = response


def get_access_token(code: str) -> GithubAccessCode:
    result = github_post("/login/oauth/access_token", code=code)
    if 200 <= result.status_code < 400:
        return GithubAccessCode.construct(**result.json())
    raise InvalidGithubCode()


def get_user(
    access_token: str,
) -> GithubUser:
    result = github_get(url=GITHUB_API_URL, path="user", access_token=access_token)
    if 200 <= result.status_code < 400:
        github_user = GithubUser.construct(**result.json())
        if not github_user.email:
            email_response = github_get(
                url=GITHUB_API_URL, path="user/emails", access_token=access_token
            )
            assert (
                200 <= email_response.status_code < 400
            ), "Get user's email from github failed'"
            email_json = email_response.json()
            primary_email = None
            for email in email_json:
                if email["verified"] and email["primary"]:
                    primary_email = email["email"]
            assert primary_email is not None, "User has no valid github visible email"
            github_user.email = primary_email
        return github_user
    else:
        error(result.json())
        raise GithubFailure("failed to get result", response=result)
