from sqlalchemy.orm import relationship
from sqlalchemy.sql.expression import and_
from sqlalchemy.sql.functions import current_timestamp
from sqlalchemy.sql.schema import Column, ForeignKey, ForeignKeyConstraint
from sqlalchemy.sql.sqltypes import JSON, DateTime, Integer, String

from app.db.base_class import Base
from app.features.model.model import ModelVersion


class Experiment(Base):
    """Database mapper for the experiment entity
    Attributes:
        stage (Column): (class attribute) one of "STARTED" "FAILED" "SUCCESS"
            "RUNNING" "NOT RUNNING"
        history (Column): (class attribute) Metrics per epoch
        hyperparams (Column): (class attribute) All network hyperparams used
            in the training
        epochs (Column): (class attribute) Number of epochs on training experiment
    """

    experiment_name = Column(String)
    model_version_name = Column(String)
    model_name = Column(String)
    created_at = Column(DateTime, server_default=current_timestamp())
    created_by_id = Column(Integer, ForeignKey("user.id", ondelete="SET NULL"))
    updated_at = Column(DateTime, server_default=current_timestamp())
    experiment_id = Column(String, primary_key=True)
    stage = Column(String, server_default="RUNNING")
    history = Column(JSON)
    epochs = Column(Integer)
    train_metrics = Column(JSON)
    val_metrics = Column(JSON)
    test_metrics = Column(JSON)

    hyperparams = Column(JSON)

    created_by = relationship(
        "User",
    )
    model_version = relationship(
        "ModelVersion",
        # back_populates="experiments",
        primaryjoin=and_(
            ModelVersion.model_name == model_name,
            ModelVersion.model_version == model_version_name,
        ),
        foreign_keys=[model_version_name, model_name],
    )
    __table_args__ = (
        ForeignKeyConstraint(
            ["model_name", "model_version_name"],
            ["modelversion.model_name", "modelversion.model_version"],
            ondelete="CASCADE",
        ),
    )
