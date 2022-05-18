from humps import camel
from pydantic.main import BaseModel


class ApiBaseModel(BaseModel):
    class Config:
        alias_generator = camel.case
        allow_population_by_field_name = True
        orm_mode = True
