from typing import Optional

from humps import camel
from pydantic import EmailStr
from pydantic.main import BaseModel


# Shared properties
class UserBase(BaseModel):
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = True
    is_superuser: bool = False
    full_name: Optional[str] = None

    class Config:
        alias_generator = camel.case
        allow_population_by_field_name = True
        orm_mode = True


class UserInDBBase(UserBase):
    id: Optional[int] = None

    class Config:
        orm_mode = True


class UserCreateBasic(UserBase):
    email: EmailStr
    password: str


class UserCreateOAuth(UserBase):
    email: EmailStr


# Properties to receive via API on update
class UserUpdate(UserBase):
    password: Optional[str] = None


# Additional properties to return via API
class User(UserInDBBase):
    pass


# Additional properties stored in DB
class UserInDB(UserInDBBase):
    hashed_password: str
