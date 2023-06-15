"""
Data type schemas used to describe datasets.
"""
from typing import Any, Literal, Union

from humps import camel
from pydantic import BaseModel, Field, validator


class BaseDataType(BaseModel):
    """
    Base pydantic model data type.
    """

    class Config:
        """Configures the wrapper class to work as intended."""

        alias_generator = camel.case
        allow_population_by_field_name = True
        allow_population_by_alias = True
        underscore_attrs_are_private = True


class NumericDataType(BaseDataType):
    """
    Data type for a numerical series.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.domain_kind = "numeric"

    domain_kind: Literal["numeric"] = Field("numeric")

    @validator("domain_kind")
    def check_domain_kind(cls, _value: Any):
        """Validates domain_kind"""
        return "numeric"


class QuantityDataType(NumericDataType):
    """
    Data type for a numerical series bound to a unit
    """

    unit: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.domain_kind = "numeric"


class StringDataType(BaseDataType):
    """
    Data type for series of strings
    """

    domain_kind: Literal["string"] = Field("string")

    @validator("domain_kind")
    def check_domain_kind(cls, _value: Any):
        """Validates domain_kind"""
        return "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.domain_kind = "string"


class CategoricalDataType(BaseDataType):
    """
    Data type for a series of categorical column
    """

    domain_kind: Literal["categorical"] = Field("categorical")
    classes: dict[Union[str, int], int]

    @validator("domain_kind")
    def check_domain_kind(cls, _value: Any):
        """Validates domain_kind"""
        return "categorical"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.domain_kind = "categorical"


class SmileDataType(BaseDataType):
    """
    Data type for a series of SMILEs strings column
    """

    domain_kind: Literal["smiles"] = Field("smiles")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.domain_kind = "smiles"

    @validator("domain_kind")
    def check_domain_kind(cls, _value: Any):
        """Validates domain_kind"""
        return "smiles"


class DNADataType(BaseDataType):
    """
    Data type for a series of DNA strings column
    """

    domain_kind: Literal["dna"] = Field("dna")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.domain_kind = "dna"

    @validator("domain_kind")
    def check_domain_kind(cls, _value: Any):
        """Validates domain_kind"""
        return "dna"


class RNADataType(BaseDataType):
    """
    Data type for a series of RNA strings column
    """

    domain_kind: Literal["rna"] = Field("rna")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.domain_kind = "rna"

    @validator("domain_kind")
    def check_domain_kind(cls, _value: Any):
        """Validates domain_kind"""
        return "rna"


class ProteinDataType(BaseDataType):
    """
    Data type for a series of protein strings column
    """

    domain_kind: Literal["protein"] = Field("protein")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.domain_kind = "protein"

    @validator("domain_kind")
    def check_domain_kind(cls, _value: Any):
        """Validates domain_kind"""
        return "protein"


DataType = Union[
    NumericDataType,
    CategoricalDataType,
    QuantityDataType,
    SmileDataType,
    DNADataType,
    RNADataType,
    ProteinDataType,
]
