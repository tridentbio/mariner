from sqlalchemy import JSON, Column, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql.functions import current_timestamp
from sqlalchemy.sql.schema import ForeignKey
from sqlalchemy.sql.sqltypes import DateTime

from app.db.base_class import Base


class ModelVersion(Base):
    model_name = Column(
        String, ForeignKey("model.name", ondelete="CASCADE"), primary_key=True
    )
    # model = relationship("Model", back_populates="versions")
    model_version = Column(String, primary_key=True)
    # experiments = relationship("Experiment", back_populates="model_version")
    config = Column(JSON)
    created_at = Column(DateTime, server_default=current_timestamp())
    updated_at = Column(DateTime, server_default=current_timestamp())



class ModelFeaturesAndTarget(Base):
    model_name = Column(
        String, ForeignKey("model.name", ondelete="CASCADE"), primary_key=True
    )
    column_name = Column(String, primary_key=True)
    column_type = Column(String)


class Model(Base):
    name = Column(String, primary_key=True)
    created_by_id = Column(Integer, ForeignKey("user.id", ondelete="SET NULL"))
    created_by = relationship(
        "User",
    )
    dataset_id = Column(Integer, ForeignKey("dataset.id", ondelete="SET NULL"))

    dataset = relationship("Dataset")
    versions = relationship("ModelVersion", cascade="all,delete")
    columns = relationship("ModelFeaturesAndTarget", cascade="all,delete")
    created_at = Column(DateTime, server_default=current_timestamp())
    updated_at = Column(DateTime, server_default=current_timestamp())
