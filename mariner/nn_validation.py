"""
Helps model building providing validation to needed
python/neural-networks queries, such as if a type
matches against some other type or 2 matrix shapes 
are multipliable


Since this should be a lightweight module, it would
be best to have it separate apart later, in such a way
that is possible to load it only with it's dependencies
"""

from typing import List

import typeguard
from humps import camel
from pydantic import BaseModel, ValidationError, validator


class CheckerBaseModel(BaseModel):
    class Config:
        alias_generator = camel.case
        allow_population_by_field_name = True
        allow_population_by_alias = True
        orm_mode = True
        underscore_attrs_are_private = True


class CheckTypeHints(CheckerBaseModel):
    types: List[str]
    expected_type: str

    @validator("types")
    def check_valid_types(self, value: List[str], values):
        for type in values.types:
            # TODO: check type against expected type
            ...
