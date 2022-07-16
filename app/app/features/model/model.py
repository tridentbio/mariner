from sqlalchemy import JSON, Column, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql.schema import ForeignKey

from app.db.base_class import Base


class ModelVersion(Base):
    model_name = Column(
        String, ForeignKey("model.name", ondelete="CASCADE"), primary_key=True
    )
    model_version = Column(String, primary_key=True)
    config = Column(JSON)


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
    dataset_id = Column(Integer, ForeignKey("dataset.id", ondelete="CASCADE"))

    dataset = relationship("Dataset")
    versions = relationship("ModelVersion")
    columns = relationship("ModelFeaturesAndTarget")
