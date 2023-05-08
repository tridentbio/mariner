"""
Experiment entity
"""

from sqlalchemy.orm import relationship
from sqlalchemy.sql.functions import current_timestamp
from sqlalchemy.sql.schema import Column, ForeignKey
from sqlalchemy.sql.sqltypes import JSON, DateTime, Integer, String

from mariner.db.base_class import Base
from mariner.entities.model import ModelVersion
from mariner.entities.user import User


class Experiment(Base):
    """Database mapper for the experiment entity
    Attributes:
        id: Integer identification of the experiment
        experiment_name: Optional name for the experiment
        mlflow_id: String that maps one instance of this entity to mlflow's equivalent
        stage (Column): (class attribute) one of "STARTED" "FAILED" "SUCCESS"
        model_version_id: Id of the model version used in the experiment
        created_at (Column): Timestamp of creation of this entity
        updated_at (Column): Timestamp of the last mutation of this entity
            "RUNNING" "NOT RUNNING"
        history (Column): (class attribute) Metrics per epoch
        hyperparams (Column): (class attribute) All network hyperparams used
            in the training
        epochs (Column): (class attribute) Number of epochs on training experiment
        train_metrics (Column): JSON string with the training metrics
        val_metrics(Column): JSON string with the validation metrics
        test_metrics(Column): JSON string with the test metrics

        created_by: User entity that created the experiment
        model_version: ModelVersion entity identified my `model_version_id`
    """

    id = Column(Integer, primary_key=True, unique=True, index=True)
    experiment_name = Column(String, nullable=False)
    mlflow_id = Column(String, nullable=True)
    model_version_id = Column(
        Integer, ForeignKey("modelversion.id", ondelete="CASCADE"), nullable=False
    )
    created_at = Column(DateTime, server_default=current_timestamp())
    created_by_id = Column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=False
    )
    updated_at = Column(DateTime, server_default=current_timestamp())
    stage = Column(String, server_default="RUNNING", nullable=False)
    epochs = Column(Integer, nullable=False)
    history = Column(JSON)
    train_metrics = Column(JSON)
    val_metrics = Column(JSON)
    test_metrics = Column(JSON)
    hyperparams = Column(JSON)
    stack_trace = Column(String)

    created_by = relationship(
        User,
    )
    model_version = relationship(
        ModelVersion,
    )
