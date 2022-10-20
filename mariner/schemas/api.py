from datetime import datetime, timezone
from typing import Generic, List, TypeVar
from humps import camel
from pydantic.generics import GenericModel
from pydantic.main import BaseModel
from pydantic.datetime_parse import parse_datetime


class ApiBaseModel(BaseModel):
    class Config:
        alias_generator = camel.case
        allow_population_by_field_name = True
        allow_population_by_alias = True
        orm_mode = True
        underscore_attrs_are_private = True


DataT = TypeVar("DataT")


class Paginated(GenericModel, Generic[DataT]):
    data: List[DataT]
    total: int


class PaginatedApiQuery(ApiBaseModel):
    page: int = 0
    per_page: int = 15


class utc_datetime(datetime):
    @classmethod
    def __get_validators__(cls):
        yield parse_datetime
        yield cls.ensure_tzinfo

    @classmethod
    def ensure_tzinfo(cls, v):
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    @staticmethod
    def to_str(dt: datetime) -> str:
        return dt.isoformat()
