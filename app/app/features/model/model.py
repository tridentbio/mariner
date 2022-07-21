from sqlalchemy import JSON, Column, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql.expression import and_
from sqlalchemy.sql.functions import current_timestamp
from sqlalchemy.sql.schema import ForeignKey, ForeignKeyConstraint
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


class Experiment(Base):
    model_version_name = Column(String)
    model_name = Column(String)
    model_version = relationship(
        "ModelVersion",
        # back_populates="experiments",
        primaryjoin=and_(
            ModelVersion.model_name == model_name,
            ModelVersion.model_version == model_version_name,
        ),
        foreign_keys=[model_version_name, model_name],
    )
    created_at = Column(DateTime, server_default=current_timestamp())
    updated_at = Column(DateTime, server_default=current_timestamp())
    experiment_id = Column(String, primary_key=True)
    stage = Column(String, server_default="RUNNING")
    __table_args__ = (
        ForeignKeyConstraint(
            ["model_name", "model_version_name"],
            ["modelversion.model_name", "modelversion.model_version"],
            ondelete="CASCADE"
        ),
    )


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
