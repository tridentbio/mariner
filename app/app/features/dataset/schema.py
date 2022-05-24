import json
from datetime import datetime
from json.decoder import JSONDecodeError
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi.datastructures import UploadFile
from pydantic.main import BaseModel

from app.schemas.api import ApiBaseModel

SplitType = Literal["scaffold", "random"]
ColumnType = Literal["numerical", "categorical", "string"]


class Split(str):
    train_percents: int
    test_percents: int
    val_percents: int

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def _modify_schema__(cls, field_schema):
        field_schema.update(examples=["60-20-20", "70-20-10"])

    @classmethod
    def validate(cls, v):
        try:
            splits = [int(s) for s in v.split("-")]
            total = sum(splits)
            if total != 100:
                raise ValueError("splits should sum to 100")
            return cls(v)
        except ValueError:
            raise ValueError('Split should be string "int-int-int"')


class DatasetsQuery(ApiBaseModel):
    sort_by_rows: Optional[str]
    sort_by_cols: Optional[str]
    sort_by_created_at: Optional[str]
    page: int = 1
    per_page: int = 15
    search_by_name: Optional[str]
    created_by_id: Optional[int]


# class DatasetStats(ApiBaseModel):
#    min: Optional[float]
#    max: Optional[float]
#    avg: Optional[float]
#    na_count: Optional[float]
#    type: ColumnType
#    std_dev: Optional[float]
DatasetStats = Any


class ColumnDescription(BaseModel):
    pattern: str
    description: str
    dataset_id: Optional[int] = None


class ColumnDescriptionFromJSONStr(str):
    pattern: str
    description: str

    def __init__(self, pattern: str, description: str):
        self.pattern = pattern
        self.description = description

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def _modify_schema__(cls, field_schema):
        field_schema.update(examples=["60-20-20", "70-20-10"])

    @classmethod
    def validate(cls, v):
        try:
            encoded = json.loads(v)
            if isinstance(encoded, list):
                return [cls(d["pattern"], d["description"]) for d in encoded]
            if "pattern" not in encoded or "description" not in encoded:
                raise ValueError("Should have pattern and description")
            pattern = encoded["pattern"]
            description = encoded["description"]
            return cls(pattern, description)
        except JSONDecodeError:
            raise ValueError("Should be a json")


class DatasetBase(ApiBaseModel):
    name: str
    description: str
    rows: int
    columns: int
    bytes: int
    stats: DatasetStats
    data_url: str
    split_target: Split
    split_actual: Optional[Split]
    split_type: SplitType
    created_at: datetime
    created_by_id: int
    columns_descriptions: Optional[List[ColumnDescription]] = None


class ColumnsMeta(BaseModel):
    name: str
    nacount: int
    dtype: str


class DatasetCreate(BaseModel):
    file: UploadFile
    name: str
    description: str
    split_target: Split
    split_type: SplitType = "random"
    columns_descriptions: List[ColumnDescriptionFromJSONStr] = []


class DatasetCreateRepo(DatasetBase):
    pass


class Dataset(DatasetBase):
    id: int


class DatasetUpdate(ApiBaseModel):
    file: Optional[UploadFile]
    name: Optional[str]
    description: Optional[str]
    split_target: Optional[Split]
    split_type: Optional[SplitType] = "random"


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
    split_type: Optional[SplitType] = None
