"""
Security service
"""
from datetime import datetime, timedelta
from typing import Any, Optional, Union

from jose import jwt
from passlib.context import CryptContext

from mariner.core.config import settings
from mariner.schemas.token import TokenPayload

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ALGORITHM = "HS256"


def create_access_token(
    subject: Union[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Creates an access token.

    Should only be used after confirming an user's identity.

    Args:
        subject:
        expires_delta:

    Returns:
        Authenticated JWT string.
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(
        to_encode, settings.AUTHENTICATION_SECRET_KEY, algorithm=ALGORITHM
    )
    return encoded_jwt


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies if a password matched it's hashed version from the users table.

    Args:
        plain_password: Plain password string to check.
        hashed_password: String with correct password hash.

    Returns:
        True if plain_password is correct, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hashes a password."""
    return pwd_context.hash(password)


def generate_deployment_signed_url(sub: Union[str, Any]) -> str:
    """Generates a signed URL for a route.
    Signed url make possible to get a resource without authentication.

    Args:
        sub: deployment_id.

    Returns:
        Signed URL.
    """
    encoded_jwt = jwt.encode(
        {"sub": str(sub)},
        settings.DEPLOYMENT_URL_SIGNATURE_SECRET_KEY,
        algorithm=ALGORITHM,
    )
    token = encoded_jwt.replace(
        ".", "/"
    )  # replace . with / to avoid problems with browser
    return f"{settings.WEBAPP_URL}/public-model/{token}"


def decode_deployment_url_token(token: str):
    """Decodes a signed URL token for a public deployment.

    Token should be valid and signed by the deployment signature.
    Token should have deployment_id as subject.

    Returns:
        TokenPayload with deployment_id as subject.

    Raises:
        JWTError: If token is invalid.
        ValidationError: If token payload is invalid.
    """
    payload = jwt.decode(
        token, settings.DEPLOYMENT_URL_SIGNATURE_SECRET_KEY, algorithms=[ALGORITHM]
    )
    token_data = TokenPayload(**payload)
    return token_data
