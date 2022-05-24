from typing import TYPE_CHECKING

from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql.schema import ForeignKey
from sqlalchemy.sql.sqltypes import JSON, DateTime

from app.db.base_class import Base

if TYPE_CHECKING:
    from .user import User  # noqa: F401


class ColumnDescription(Base):
    pattern = Column(String, primary_key=True)
    description = Column(String)
    dataset_id = Column(Integer, ForeignKey("dataset.id"), primary_key=True)
    dataset = relationship("Dataset", back_populates="columns_descriptions")


class Dataset(Base):
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)
    bytes = Column(Integer)
    rows = Column(Integer)
    stats = Column(JSON)
    split_target = Column(String)
    split_actual = Column(String)
    split_type = Column(String)
    data_url = Column(String)
    columns = Column(Integer)
    created_at = Column(DateTime)
    created_by_id = Column(Integer, ForeignKey("user.id"))
    created_by = relationship("User", back_populates="datasets")
    columns_descriptions = relationship(
        "ColumnDescription", back_populates="dataset", cascade="all,delete"
    )
