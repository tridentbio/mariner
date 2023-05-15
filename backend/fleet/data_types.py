"""
Data type schemas used to describe datasets.
"""
from typing import Any, Literal, Union

from pydantic import Field, validator

from fleet.model_builder.utils import CamelCaseModel


class NumericalDataType(CamelCaseModel):
    """
    Data type for a numerical series
    """

    domain_kind: Literal["numeric"] = Field("numeric")

    @validator("domain_kind")
    def check_domain_kind(cls, value: Any):
        """Validates domain_kind"""
        return "numeric"


class QuantityDataType(NumericalDataType):
    """
    Data type for a numerical series bound to a unit
    """

    unit: str


class StringDataType(CamelCaseModel):
    """
    Data type for series of strings
    """

    domain_kind: Literal["string"] = Field("string")

    @validator("domain_kind")
    def check_domain_kind(cls, value: Any):
        """Validates domain_kind"""
        return "string"


class CategoricalDataType(CamelCaseModel):
    """
    Data type for a series of categorical column
    """

    domain_kind: Literal["categorical"] = Field("categorical")
    classes: dict[Union[str, int], int]

    @validator("domain_kind")
    def check_domain_kind(cls, value: Any):
        """Validates domain_kind"""
        return "categorical"


class SmileDataType(CamelCaseModel):
    """
    Data type for a series of SMILEs strings column
    """

    domain_kind: Literal["smiles"] = Field("smiles")

    @validator("domain_kind")
    def check_domain_kind(cls, value: Any):
        """Validates domain_kind"""
        return "smiles"


class DNADataType(CamelCaseModel):
    """
    Data type for a series of DNA strings column
    """

    domain_kind: Literal["dna"] = Field("dna")

    @validator("domain_kind")
    def check_domain_kind(cls, value: Any):
        """Validates domain_kind"""
        return "dna"


class RNADataType(CamelCaseModel):
    """
    Data type for a series of RNA strings column
    """

    domain_kind: Literal["rna"] = Field("rna")

    @validator("domain_kind")
    def check_domain_kind(cls, value: Any):
        """Validates domain_kind"""
        return "rna"


class ProteinDataType(CamelCaseModel):
    """
    Data type for a series of protein strings column
    """

    domain_kind: Literal["protein"] = Field("protein")

    @validator("domain_kind")
    def check_domain_kind(cls, value: Any):
        """Validates domain_kind"""
        return "protein"
