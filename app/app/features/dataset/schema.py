import json
from datetime import datetime
from json.decoder import JSONDecodeError
from typing import Any, Dict, List, Literal, Optional, Union, no_type_check

from fastapi.datastructures import UploadFile
from pydantic.main import BaseModel
from sqlalchemy.sql.sqltypes import Enum

from app.schemas.api import ApiBaseModel, PaginatedApiQuery

SplitType = Literal["scaffold", "random"]


class Split(str):
    string: str
    train_percents: int
    test_percents: int
    val_percents: int

    def __init__(self, string: str):
        self.string = string

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def _modify_schema__(cls, field_schema):
        field_schema.update(examples=["60-20-20", "70-20-10"])

    @classmethod
    def build(
        cls, *, train_percents: int, test_percents: int, val_percents: int
    ) -> str:
        return f"{train_percents}-{test_percents}-{val_percents}"

    @no_type_check
    def __new__(cls, url: Optional[str], **kwargs) -> object:
        return str.__new__(cls, cls.build(**kwargs) if url is None else url)

    @classmethod
    def validate(cls, v):
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
    sort_by_rows: Optional[str]
    sort_by_cols: Optional[str]
    sort_by_created_at: Optional[str]
    search_by_name: Optional[str]
    created_by_id: Optional[int]


class DataType(str, Enum):
    string = "string"
    numerical = "numerical"
    smiles = "smiles"
    categorical = "categorical"


class ColumnMetadata(ApiBaseModel):
    data_type: str
    description: str
    pattern: str
    dataset_id: Optional[int] = None


class ColumnMetadataFromJSONStr(str):
    metadatas: List[ColumnMetadata] = []

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        try:
            encoded = json.loads(v)
            assert isinstance(encoded, list), "should be array"
            metadatas = []
            for obj in encoded:
                metadatas.append(ColumnMetadata.parse_obj(obj))
            instance = cls(v)
            instance.metadatas = metadatas
            return instance
        except JSONDecodeError:
            raise ValueError("Should be a json string")


class DatasetBase(ApiBaseModel):
    name: str
    description: str
    rows: int
    columns: int
    bytes: int
    stats: Any
    data_url: str
    split_target: Split
    split_actual: Optional[Split]
    split_type: SplitType
    created_at: datetime
    updated_at: datetime
    created_by_id: int
    columns_metadata: List[ColumnMetadata] = []


class ColumnsMeta(BaseModel):
    name: str
    nacount: int
    dtype: str


class DatasetCreate(BaseModel):
    file: UploadFile
    name: str
    description: str
    split_target: Split
    split_column: str = None
    split_type: SplitType = "random"
    columns_metadata: List[ColumnMetadata] = []


class DatasetCreateRepo(DatasetBase):
    pass


class Dataset(DatasetBase):
    id: int


class DatasetUpdate(ApiBaseModel):
    file: Optional[UploadFile] = None
    name: Optional[str] = None
    description: Optional[str] = None
    split_column: Optional[str] = None
    split_target: Optional[Split] = None
    split_type: Optional[SplitType] = None
    columns_metadata: Optional[List[ColumnMetadata]] = None


class DatasetUpdateRepo(BaseModel):
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
    columns_metadata: Optional[List[ColumnMetadata]] = None
