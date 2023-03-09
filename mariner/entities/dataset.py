"""
Dataset entity and core relations

Core relations include: (1-N) ColumnMetadata describing set of columns
"""
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql.functions import current_timestamp
from sqlalchemy.sql.schema import ForeignKey
from sqlalchemy.sql.sqltypes import JSON, DateTime

from mariner.db.base_class import Base


class ColumnsMetadata(Base):
    """Entity mapping column information provided by user during dataset creation."""

    data_type = Column(JSON, nullable=False)
    description = Column(String, nullable=True)
    pattern = Column(String, primary_key=True)
    dataset_id = Column(
        Integer, ForeignKey("dataset.id", ondelete="CASCADE"), primary_key=True
    )
    dataset = relationship(
        "Dataset",
        back_populates="columns_metadata",
    )


class Dataset(Base):
    """Entity mapping to dataset created by users."""

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, unique=True)
    description = Column(String)
    bytes = Column(Integer, nullable=True)
    rows = Column(Integer)
    stats = Column(JSON)
    split_target = Column(String)
    split_actual = Column(String, nullable=True)
    split_type = Column(String)
    split_column = Column(String, nullable=True)
    data_url = Column(String, nullable=True)
    columns = Column(Integer)
    created_at = Column(DateTime, server_default=current_timestamp())
    updated_at = Column(DateTime, server_default=current_timestamp())
    created_by_id = Column(Integer, ForeignKey("user.id"))
    created_by = relationship("User", back_populates="datasets")
    columns_metadata = relationship(
        "ColumnsMetadata", back_populates="dataset", cascade="all,delete"
    )
    ready_status = Column(String, nullable=True)
    errors = Column(JSON, nullable=True)

    def get_dataframe(self):
        """Loads the dataframe linked to this dataset from s3.

        Gets dataframe stored in this dataset data_url attribute at
        datasets bucket.
        """
        from mariner.core.aws import Bucket, download_file_as_dataframe

        df = download_file_as_dataframe(Bucket.Datasets, self.data_url)
        return df
