"""
Authentication related Data Transfer Object
"""

from typing import Optional

from pydantic import BaseModel


class Token(BaseModel):
    """Payload to return when an user successfully authenticates."""

    access_token: str
    token_type: str


class TokenPayload(BaseModel):
    sub: Optional[int] = None
