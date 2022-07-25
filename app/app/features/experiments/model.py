from sqlalchemy.orm import relationship
from sqlalchemy.sql.expression import and_
from sqlalchemy.sql.functions import current_timestamp
from sqlalchemy.sql.schema import Column, ForeignKeyConstraint
from sqlalchemy.sql.sqltypes import DateTime, String

from app.db.base_class import Base
from app.features.model.model import ModelVersion


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
            ondelete="CASCADE",
        ),
    )
