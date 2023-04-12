"""
User related Data Transfer Objects.
"""
from typing import Optional

from pydantic import EmailStr

from mariner.schemas.api import ApiBaseModel


# Shared properties
class UserBase(ApiBaseModel):
    """Base payload when receiving or returning a user."""

    email: Optional[EmailStr] = None
    is_active: Optional[bool] = True
    is_superuser: bool = False
    full_name: Optional[str] = None


class UserInDBBase(UserBase):
    """Models a user persisted as returned by data layer."""

    id: Optional[int] = None

    class Config:
        orm_mode = True


class UserCreateBasic(UserBase):
    """Payload for creating a user through basic authentication.

    Attributes:
        email: email to link the account.
        password: password for basic authentication."""

    email: EmailStr
    password: str


class UserCreateOAuth(UserBase):
    """Payload for creating a user through oauth."""

    image_url: Optional[str]
    email: EmailStr


# Properties to receive via API on update
class UserUpdate(UserBase):
    """Payload for updating user."""

    password: Optional[str] = None


# Additional properties to return via API
class User(UserInDBBase):
    """User represents the account information unique to each user of the application.

    Attributes:
        id: unique id of user.
        email: user email.
        is_active: flag to control user access.
        is_superuser: flag to mark user as superuser.
        full_name: name of the user.
    """


# Additional properties stored in DB
class UserInDB(UserInDBBase):
    """User complete information including password hash.

    Should not use this model on endpoint definitions.
    """

    hashed_password: str
