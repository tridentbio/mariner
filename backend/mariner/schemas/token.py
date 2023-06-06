"""
Authentication related Data Transfer Object
"""

from typing import Optional

from pydantic import BaseModel, validator


class Token(BaseModel):
    """Payload to return when an user successfully authenticates."""

    access_token: str
    token_type: str


class TokenPayload(BaseModel):
    sub: Optional[int] = None

    @validator("sub")
    def sub_cast_validator(cls, v):
        """Converts sub to int if it's a string.

        Raises:
            ValueError: If sub is not a string or int.
        """
        if isinstance(v, str):
            return int(v)
        return v
