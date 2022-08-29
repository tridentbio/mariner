from typing import Any, Dict, List, Literal, Optional

import pydantic
from sqlalchemy.orm.session import Session

from app.crud.base import CRUDBase
from app.features.experiments.model import Experiment
from app.features.model.model import ModelVersion


class ExperimentCreateRepo(pydantic.BaseModel):
    mlflow_id: str
    model_version_id: int
    created_by_id: int
    epochs: int
    experiment_name: Optional[str] = None
    stage: Literal[
        "NOT RUNNING", "STARTED", "RUNNING", "FAILED", "SUCCESS"
    ] = "NOT RUNNING"
    train_metrics: Optional[Dict[str, float]] = None
    hyperparams: Optional[Dict[str, Any]] = None
    val_metrics: Optional[Dict[str, float]] = None
    test_metrics: Optional[Dict[str, float]] = None
    history: Optional[Dict[str, List[float]]] = None


class ExperimentUpdateRepo(pydantic.BaseModel):
    stage: Optional[
        Literal["NOT RUNNING", "STARTED", "RUNNING", "FAILED", "SUCCESS"]
    ] = None
    train_metrics: Optional[Dict[str, float]] = None
    hyperparams: Optional[Dict[str, Any]] = None
    val_metrics: Optional[Dict[str, float]] = None
    test_metrics: Optional[Dict[str, float]] = None
    history: Optional[Dict[str, List[float]]] = None
    epochs: Optional[int] = None
    stack_trace: Optional[str] = None


class CRUDExperiment(CRUDBase[Experiment, ExperimentCreateRepo, ExperimentUpdateRepo]):
    def get_by_model_id(self, db: Session, id: int):
        return (
            db.query(Experiment)
            .join(Experiment.model_version)
            .filter(ModelVersion.model_id == id)
            .all()
        )

    def get(self, db: Session, experiment_id: int):
        return db.query(Experiment).filter(Experiment.id == experiment_id).first()


repo = CRUDExperiment(Experiment)
