"""
API related Data Transfer Objects.
"""
from datetime import datetime, timezone
from typing import Any, Generic, List, Literal, TypeVar, Union
from urllib.parse import unquote

from fastapi import Query
from humps import camel
from pydantic.datetime_parse import parse_datetime
from pydantic.generics import GenericModel
from pydantic.main import BaseModel


class ApiBaseModel(BaseModel):
    """Configures all payloads inputs and outputs of the domain layer
    to be convertible to and from camelCase.
    """

    class Config:
        """Configures pydantic behaviour to work as intended."""

        alias_generator = camel.case
        allow_population_by_field_name = True
        allow_population_by_alias = True
        orm_mode = True
        underscore_attrs_are_private = True

    @classmethod
    def from_orm_array(cls, entities: List[Any]):
        """Parses an array of entities to the calling class pydantic model.

        Args:
            entities: Array of entity instances to be parsed by pydantic model.
        """
        return [cls.from_orm(entity) for entity in entities]


DataT = TypeVar("DataT")


class Paginated(GenericModel, Generic[DataT]):
    """Models a generic paginated payload data."""

    data: List[DataT]
    total: int


class PaginatedApiQuery(ApiBaseModel):
    """Models a paginated query of a resource."""

    page: int = 0
    per_page: int = 15


class utc_datetime(datetime):
    """Class converts datetime field to utc format to let the client
    display a localized version of the same timestamp."""

    @classmethod
    def __get_validators__(cls):
        yield parse_datetime
        yield cls._ensure_tzinfo

    @classmethod
    def _ensure_tzinfo(cls, v):
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)


class OrderByClause(ApiBaseModel):
    """Class to model an ordering over a query."""

    field: str
    order: Literal["asc", "desc"]


class OrderByQuery(ApiBaseModel):
    """Models a complex sorting on multiple columns."""

    clauses: List[OrderByClause]


def get_order_by_query(
    order_by: Union[str, None] = Query(
        default=None,
        alias="orderBy",
        description="Describes how the query is to be sorted",
        examples={
            "normal": {
                "summary": "Example of orderBy usage",
                "description": (
                    "Order is applied by createdAt in ascending order and"
                    "then by name on descending order if createdAt are the same"
                ),
                "value": "+createdAt,-name",
            },
        },
    )
):
    """Maps the order_by query string into pydantic model describing the
    requested order.

    Splits the querystring by comma and maps elements to ordering clause
    :class:`OrderByQuery`.

    Attributes:
        order_by: string to be parsed (gotten by fastapi from the request's
                                       querystring order_by or ``orderBy``.
    """
    if not isinstance(order_by, str):
        return None
    result = []
    order_by = unquote(order_by)

    for order_clause in order_by.split(","):
        field = order_clause[1:]
        if order_clause.startswith("+"):
            result.append(OrderByClause.construct(field=field, order="asc"))
        else:
            result.append(OrderByClause.construct(field=field, order="desc"))

    return OrderByClause.construct(clauses=result)