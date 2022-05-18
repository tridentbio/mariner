from datetime import datetime
from enum import Enum
from typing import Dict, Literal, Optional

from fastapi.datastructures import UploadFile
from pydantic.main import BaseModel

SplitType = Literal['scaffold', 'random']

class Split(str):
    train_percents: int
    test_percents: int
    val_percents: int

    @classmethod 
    def __get_validators__(cls):
        yield cls.validate
    @classmethod 
    def _modify_schema__(cls, field_schema):
        field_schema.update(
            examples=["60-20-20", "70-20-10"]
        )
    @classmethod
    def validate(cls, v):
        try:
            splits = [int(s) for s in v.split('-')]
            total = sum(splits)
            if total != 100:
                raise ValueError('splits should sum to 100')
            return cls(v)
        except:
            raise ValueError('Split should be string "int-int-int"')

class DatasetsQuery(BaseModel):
    sort_by_rows: Optional[str]
    sort_by_cols: Optional[str]
    sort_by_created_at: Optional[str] 
    page: int = 1 
    per_page: int = 15

    search_by_name: Optional[str]
    created_by_id: Optional[int]

# Shared properties
class DatasetBase(BaseModel):
    name: str
    description: str
    rows: int
    columns: int
    bytes: int
    stats: Dict
    data_url: str
    split_target: Split
    split_actual: Optional[Split]
    split_type: SplitType
    created_at: datetime
    created_by_id: int

    class Config:
        orm_mode = True
    
class DatasetCreate(BaseModel):
    file: UploadFile
    name: str
    description: str
    split_target: Split
    split_type: SplitType = 'random'

class DatasetCreateRepo(DatasetBase):
    pass

class Dataset(DatasetBase):
    id: int

class DatasetUpdate(BaseModel):
    name: Optional[str]
    description: Optional[str]
