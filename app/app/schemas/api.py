from humps import camel
from pydantic.main import BaseModel


class ApiBaseModel(BaseModel):
    class Config:
        alias_generator = camel.case
        allow_population_by_field_name = True
        allow_population_by_alias = True
        orm_mode = True
        underscore_attrs_are_private = True


class PaginatedApiQuery(ApiBaseModel):
    page: int = 0
    per_page: int = 15
