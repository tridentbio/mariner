"""
Dataset related DTOs
"""

import json
from json.decoder import JSONDecodeError
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    NewType,
    Optional,
    Tuple,
    Union,
    no_type_check,
)

import pandas as pd
from fastapi.datastructures import UploadFile
from pydantic import Field, validator
from pydantic.main import BaseModel

from fleet.model_builder.schemas import \
    CategoricalDataType as BuilderCategoricalDT
from fleet.model_builder.schemas import DNADataType as BuilderDNADT
from fleet.model_builder.schemas import NumericalDataType as BuilderNumericalDT
from fleet.model_builder.schemas import ProteinDataType as BuilderProteinDT
from fleet.model_builder.schemas import QuantityDataType as BuilderQuantityDT
from fleet.model_builder.schemas import RNADataType as BuilderRNADT
from fleet.model_builder.schemas import SmileDataType as BuilderSmilesDT
from fleet.model_builder.schemas import StringDataType as BuilderStringDT
from mariner.core.aws import Bucket, download_file_as_dataframe
from mariner.schemas.api import ApiBaseModel, PaginatedApiQuery, utc_datetime

SplitType = Literal["scaffold", "random"]

StatsType = NewType(
    "StatsType",
    Dict[
        Literal["full", "train", "test", "val"],
        Dict[str, Union[pd.Series, Dict[str, pd.Series]]],
    ],
)

SchemaType = NewType(
    "SchemaType",
    Dict[
        str,
        Union[Tuple[Callable[..., bool], str], List[Tuple[Callable[..., bool], str]]],
    ],
)


class Split(str):
    """Split class. This class is used to validate and type the split string.

    Attributes:
        string (str): split string
        train_percents (int): train percents
        test_percents (int): test percents
        val_percents (int): val percents
    """

    string: str
    train_percents: int
    test_percents: int
    val_percents: int

    def __init__(self, string: str):
        self.string = string

    @classmethod
    def __get_validators__(cls):
        yield cls._validate

    @classmethod
    def _modify_schema__(cls, field_schema):
        field_schema.update(examples=["60-20-20", "70-20-10"])

    @classmethod
    def _build(
        cls, *, train_percents: int, test_percents: int, val_percents: int
    ) -> str:
        return f"{train_percents}-{test_percents}-{val_percents}"

    @no_type_check
    def __new__(cls, url: Optional[str], **kwargs) -> object:
        return str.__new__(cls, cls._build(**kwargs) if url is None else url)

    @classmethod
    def _validate(cls, v):
        try:
            splits = [int(s) for s in v.split("-")]
            total = sum(splits)
            if total != 100:
                raise ValueError("splits should sum to 100")
            instance = cls(v)
            instance.train_percents = splits[0]
            instance.test_percents = splits[1]
            instance.val_percents = splits[2]
            return instance
        except ValueError:
            raise ValueError('Split should be string "int-int-int"')


class DatasetsQuery(PaginatedApiQuery):
    """Query for datasets.

    Attributes:
        sort_by_rows (Optional[str]): sort by rows
        sort_by_cols (Optional[str]): sort by cols
        sort_by_created_at (Optional[str]): sort by created at
        search_by_name (Optional[str]): search by name
        created_by_id (Optional[int]): created by id
    """

    sort_by_rows: Optional[str]
    sort_by_cols: Optional[str]
    sort_by_created_at: Optional[str]
    search_by_name: Optional[str]
    created_by_id: Optional[int]


class NumericalDataType(ApiBaseModel, BuilderNumericalDT):
    """Numerical data type."""


class QuantityDataType(ApiBaseModel, BuilderQuantityDT):
    """Quantity data type."""


class StringDataType(ApiBaseModel, BuilderStringDT):
    """String data type."""


class CategoricalDataType(BuilderCategoricalDT, ApiBaseModel):
    """Categorical data type."""


class SmileDataType(ApiBaseModel, BuilderSmilesDT):
    """Smile data type."""

    domain_kind: Literal["smiles"] = Field("smiles")

    @validator("domain_kind")
    def check_domain_kind(cls, v):
        return "smiles"


class DNADataType(ApiBaseModel, BuilderDNADT):
    """DNA data type."""

    domain_kind: Literal["dna"] = Field("dna")
    is_ambiguous: bool = Field(False)

    @validator("domain_kind")
    def check_domain_kind(cls, v):
        return "dna"


class RNADataType(ApiBaseModel, BuilderRNADT):
    """RNA data type."""

    domain_kind: Literal["rna"] = Field("rna")
    is_ambiguous: bool = Field(False)

    @validator("domain_kind")
    def check_domain_kind(cls, v):
        return "rna"


class ProteinDataType(ApiBaseModel, BuilderProteinDT):
    """Protein data type."""

    domain_kind: Literal["protein"] = Field("protein")

    @validator("domain_kind")
    def check_domain_kind(cls, v):
        return "protein"


BiologicalDataType = Union[DNADataType, RNADataType, ProteinDataType]


class ColumnsDescription(ApiBaseModel):
    """Columns description.

    Attributes:
        data_type (AnyDataType):
        column data type
        description (str): column description
        pattern (str): column pattern
        dataset_id (Optional[int]): dataset id
    """

    data_type: Union[
        QuantityDataType,
        StringDataType,
        CategoricalDataType,
        SmileDataType,
        BiologicalDataType,
    ] = Field(...)
    description: str
    pattern: str
    dataset_id: Optional[int] = None


class ColumnMetadataFromJSONStr(str):
    """Type for raw column metadata comming from a json request."""

    metadatas: List[ColumnsDescription] = []

    @classmethod
    def __get_validators__(cls):
        yield cls._validate

    @classmethod
    def _validate(cls, v):
        try:
            encoded = json.loads(v)
            assert isinstance(encoded, list), "should be array"
            metadatas = []
            for obj in encoded:
                metadatas.append(ColumnsDescription.parse_obj(obj))
            instance = cls(v)
            instance.metadatas = metadatas
            return instance
        except JSONDecodeError:
            raise ValueError("Should be a json string")


class DatasetBase(ApiBaseModel):
    """Dataset base type."""

    id: Optional[int] = None
    name: str
    description: str
    rows: Optional[int]
    columns: Optional[int]
    bytes: Optional[int]
    stats: Optional[Any]
    data_url: Optional[str]
    split_target: Split
    split_actual: Optional[Split]
    split_type: SplitType
    created_at: utc_datetime
    updated_at: utc_datetime
    created_by_id: int
    columns_metadata: List[ColumnsDescription] = []
    ready_status: Optional[Literal["failed", "processing", "ready"]] = "processing"
    errors: Optional[Dict[str, Union[List[str], str]]] = None


class ColumnsMeta(BaseModel):
    """Columns metadata parsed."""

    name: str
    dtype: Optional[
        Union[
            CategoricalDataType,
            NumericalDataType,
            StringDataType,
            SmileDataType,
            BiologicalDataType,
        ]
    ]


class DatasetCreate(BaseModel):
    """Dataset to create type.

    Used in the dataset creation route.
    """

    file: UploadFile
    name: str
    description: str
    split_target: Split
    split_column: Optional[str] = None
    split_type: SplitType = "random"
    columns_metadata: List[ColumnsDescription] = []


class DatasetCreateRepo(DatasetBase):
    """Dataset to create type.

    Used in the DatasetCRUD to create a dataset in the database.
    """


class Dataset(DatasetBase):
    """Dataset model type."""

    id: int

    def get_dataframe(self):
        """Loads the dataframe linked to this dataset from s3.

        Gets dataframe stored in this dataset data_url attribute at
        datasets bucket.
        """
        assert self.data_url
        df = download_file_as_dataframe(Bucket.Datasets, self.data_url)
        return df


class DatasetUpdate(ApiBaseModel):
    """Dataset to update type.

    Used in the dataset update route.
    """

    file: Optional[UploadFile] = None
    name: Optional[str] = None
    description: Optional[str] = None
    split_column: Optional[str] = None
    split_target: Optional[Split] = None
    split_type: Optional[SplitType] = None
    columns_metadata: Optional[List[ColumnsDescription]] = None


class DatasetUpdateInput(ApiBaseModel):
    """Payload of a dataset update.


    Attributes:
        name: alters dataset name to this value.
        description: alters dataset's description to this value.
        split_column: alters dataset's split_column to this value.
        split_target: alters dataset's split_target to this value.
        file: alters dataset's file to this value.
        columns_metadata: alters dataset's columns_metadata do this value.

    """

    name: Optional[str] = None
    description: Optional[str] = None
    split_column: Optional[str] = None
    split_target: Optional[Split] = None
    split_type: Optional[SplitType] = None
    file: Optional[UploadFile] = None
    columns_metadata: Optional[ColumnMetadataFromJSONStr] = None


class DatasetSummary(BaseModel):
    """Dataset summary type."""

    train: Dict[str, Any]
    val: Dict[str, Any]
    test: Dict[str, Any]
    full: Dict[str, Any]


class DatasetUpdateRepo(BaseModel):
    """Dataset to update type.

    Accepted by the DatasetCRUD to update a dataset in the database.
    """

    id: int
    name: Optional[str] = None
    description: Optional[str] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    bytes: Optional[int] = None
    stats: Optional[Union[Dict, str]] = None
    data_url: Optional[str] = None
    split_target: Optional[Split] = None
    split_actual: Optional[Split] = None
    split_column: Optional[str] = None
    split_type: Optional[SplitType] = None
    columns_metadata: Optional[List[ColumnsDescription]] = None
    ready_status: Optional[str] = None
    errors: Optional[Dict[str, Union[List[str], str]]] = None


class DatasetProcessStatusEventPayload(ApiBaseModel):
    """Dataset process status event payload type.

    Used in the dataset process to send the status of the
    dataset processing to the frontend.
    """

    dataset_id: Optional[int] = None
    message: Optional[str] = "success on dataset creation"
    dataset: Optional[Dataset] = None
