from humps import camel
from pydantic.main import BaseModel


class ApiBaseModel(BaseModel):
    class Config:
        alias_generator = camel.case
