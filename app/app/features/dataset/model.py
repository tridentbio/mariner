from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql.functions import current_timestamp
from sqlalchemy.sql.schema import ForeignKey
from sqlalchemy.sql.sqltypes import JSON, DateTime

from app.core.aws import Bucket, download_file_as_dataframe
from app.db.base_class import Base


class ColumnDescription(Base):
    pattern = Column(String, primary_key=True)
    description = Column(String)
    dataset_id = Column(Integer, ForeignKey("dataset.id"), primary_key=True)
    dataset = relationship("Dataset", back_populates="columns_descriptions")


class ColumnsMetadata(Base):
    key = Column(String, primary_key=True, nullable=False)
    dataset_id = Column(Integer, ForeignKey("dataset.id"), primary_key=True)
    dataset = relationship("Dataset", back_populates="columns_metadatas")
    data_type = Column(String, nullable=False)


class Dataset(Base):
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, unique=True)
    description = Column(String)
    bytes = Column(Integer)
    rows = Column(Integer)
    stats = Column(JSON)
    split_target = Column(String)
    split_actual = Column(String)
    split_type = Column(String)
    data_url = Column(String)
    columns = Column(Integer)
    created_at = Column(DateTime, server_default=current_timestamp())
    updated_at = Column(DateTime, server_default=current_timestamp())
    created_by_id = Column(Integer, ForeignKey("user.id"))
    created_by = relationship("User", back_populates="datasets")
    columns_descriptions = relationship(
        "ColumnDescription", back_populates="dataset", cascade="all,delete"
    )
    columns_metadatas = relationship(
        "ColumnsMetadata", back_populates="dataset", cascade="all,delete"
    )

    def get_dataframe(self):
        df = download_file_as_dataframe(Bucket.Datasets, self.data_url)
        return df
