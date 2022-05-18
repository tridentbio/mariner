from datetime import datetime
from typing import Dict, Optional
from pydantic.main import BaseModel

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
    created_at: datetime
    created_by_id: int

    class Config:
        orm_mode = True
    
class DatasetCreate(BaseModel):
    name: str
    description: str

class DatasetCreateRepo(DatasetBase):
    pass

class Dataset(DatasetBase):
    id: int

class DatasetUpdate(BaseModel):
    name: Optional[str]
    description: Optional[str]
