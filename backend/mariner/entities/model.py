"""
Model entity and core relations

Core relations include: 
    - (1-N) ModelVersion with a model architecture
    - (1-N) ModelFeaturesAndTarget with features and target view of the
    dataset related to the model
"""

from sqlalchemy import JSON, Column, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql.functions import current_timestamp
from sqlalchemy.sql.schema import ForeignKey
from sqlalchemy.sql.sqltypes import DateTime

from mariner.db.base_class import Base
from mariner.entities.dataset import Dataset
from mariner.entities.user import User


class ModelVersion(Base):
    """Entity representing a model version of some model in development.
    :
        id: Model version identification
        name: Model version string identification
        mlflow_name: String used to map this version to mlflow's ModelVersion entity
        config: JSON string containing a ModelConfig instance, describing the model's
            architecture, dataset used for input and target definitions, featurizers for
            columns of special data types,
        created_at (datetime.time): Timestamp of entity creation
        updated_at (datetime.time): Timestamp of entity mutation
    """

    id = Column(Integer, primary_key=True, index=True, unique=True)

    model_id = Column(
        Integer, ForeignKey("model.id", ondelete="CASCADE"), nullable=False
    )
    description = Column(String, nullable=True)
    model = relationship("Model", back_populates="versions")
    name = Column(String, nullable=False)
    mlflow_version = Column(String, nullable=True)
    mlflow_model_name = Column(String, nullable=False)
    # experiments = relationship("Experiment", back_populates="model_version")
    check_status = Column(String, nullable=True)
    check_stack_trace = Column(String, nullable=True)
    config = Column(JSON, nullable=False)

    created_by_id = Column(Integer, ForeignKey("user.id", ondelete="SET NULL"))
    created_by = relationship(User)
    created_at = Column(
        DateTime, server_default=current_timestamp(), nullable=False
    )
    updated_at = Column(
        DateTime, server_default=current_timestamp(), nullable=False
    )


class ModelFeaturesAndTarget(Base):
    """Entity to describe a model's features and target

    Attributes:
        model_id: Identification of the model entity
        column_name: Reference to the dataset column name
        column_type ("feature" | "target"): constant string indicating if the column
        is used as a string or a target
    """

    model_id = Column(
        Integer, ForeignKey("model.id", ondelete="CASCADE"), primary_key=True
    )
    column_name = Column(String, primary_key=True)
    column_type = Column(String)


class Model(Base):
    """Entity representing a model development flow

    Attributes:
        id: Integer model identification
        name: String model identification
        created_by_id: Id of the user that created the model
        dataset_id: Id of the dataset used for train-test-validation split
        versions (List[ModelVersion]): List of version entities
        columns List: Sets of feature and target columns
        created_at (datetime.time): Timestamp of entity creation
        updated_at (datetime.time): Timestamp of entity mutation
    """

    id = Column(Integer, primary_key=True, index=True, unique=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    mlflow_name = Column(String, nullable=False)
    created_by_id = Column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=False
    )
    created_by = relationship(
        User,
    )
    dataset_id = Column(Integer, ForeignKey("dataset.id", ondelete="SET NULL"))
    dataset = relationship(Dataset)
    versions = relationship(ModelVersion, cascade="all,delete")
    columns = relationship(
        ModelFeaturesAndTarget, cascade="all,delete", uselist=True
    )
    created_at = Column(
        DateTime, server_default=current_timestamp(), nullable=False
    )
    updated_at = Column(
        DateTime, server_default=current_timestamp(), nullable=False
    )
