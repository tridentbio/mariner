"""
Interacts with Genentech's oauth.
"""
from dataclasses import dataclass

import requests
import requests.auth
from jose import jwt
from jose.exceptions import JWKError
from pydantic import BaseModel, ValidationError


# Constants
class GenentechDecodedAccessToken(BaseModel):
    """
    A Genentech decoded access token.
    """

    scope: str
    client_id: str
    userId: str
    email: str
    exp: int


class GenentechAccessCode(BaseModel):
    """
    A Genentech access token.
    """

    access_token: str
    refresh_token: str
    token_type: str  # Bearer
    expires_in: int  # ms to expire


@dataclass
class ClientOptions:
    """
    Parameters of the Client, used to run the client on different environments
    """

    client_id: str
    client_secret: str
    redirect_uri: str
    jwks_url: str
    token_url: str


class GenentechClient:
    """
    Provides an API for interacting with Genentech's oauth.
    """

    def __init__(self, options: ClientOptions):
        self.options = options

    def get_user(self, access_token: str) -> GenentechDecodedAccessToken:
        """
        Get a user from Genentech.
        """
        claims = jwt.get_unverified_claims(access_token)
        return GenentechDecodedAccessToken.parse_obj(
            claims,
        )

    def get_access_token(
        self, code: str, grant_type: str
    ) -> GenentechAccessCode:
        """
        Get an access token from Genentech.
        """
        # Create basic auth headers with client_id and client_secret
        basic_header = requests.auth.HTTPBasicAuth(
            self.options.client_id, self.options.client_secret
        )

        # Make request
        response = requests.post(
            self.options.token_url,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
            },
            auth=basic_header,
            data={
                "grant_type": grant_type,
                "code": code,
                "redirect_uri": self.options.redirect_uri,
            },
            timeout=30,
        )
        try:
            return GenentechAccessCode.parse_obj(response.json())
        except ValidationError as exc:
            raise ValueError(response.text) from exc

    def _validate_token(self, token: str) -> None:
        """
        Validate a token.
        """
        # Get keys from JWKS url
        data = requests.get(self.options.jwks_url, timeout=5).json()

        for key in data["keys"]:
            # Attempt to verify with key
            kty = key.get("kty")
            if kty == "RSA":
                public_key = key.get("n")
                try:
                    jwt.decode(token, key=public_key)
                except JWKError:
                    pass
        raise JWKError(f"None of the keys in {self.options.jwks_url} match")

    def exchange_code(self, code: str) -> GenentechDecodedAccessToken:
        """
        Exchange a code for a GenentechUser object.
        """
        token = self.get_access_token(code, "authorization_code")
        # _validate_token(token)
        return self.get_user(token.access_token)
