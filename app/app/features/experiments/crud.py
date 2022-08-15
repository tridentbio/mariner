from typing import Any, Dict, List, Literal, Optional

import pydantic
from sqlalchemy.orm.session import Session

from app.crud.base import CRUDBase
from app.features.experiments.model import Experiment


class ExperimentCreateRepo(pydantic.BaseModel):
    model_name: str
    model_version_name: str
    experiment_id: str
    created_by_id: int
    experiment_name: Optional[str] = None
    stage: Literal[
        "NOT RUNNING", "STARTED", "RUNNING", "FAILED", "SUCCESS"
    ] = "NOT RUNNING"
    train_metrics: Optional[Dict[str, float]] = None
    hyperparams: Optional[Dict[str, Any]] = None
    val_metrics: Optional[Dict[str, float]] = None
    test_metrics: Optional[Dict[str, float]] = None
    history: Optional[Dict[str, List[float]]] = None
    epochs: Optional[int] = None


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


class CRUDExperiment(CRUDBase[Experiment, ExperimentCreateRepo, ExperimentUpdateRepo]):
    def get_by_model_name(self, db: Session, model_name: str):
        return db.query(Experiment).filter(Experiment.model_name == model_name).all()

    def get(self, db: Session, experiment_id: str):
        return (
            db.query(Experiment)
            .filter(Experiment.experiment_id == experiment_id)
            .first()
        )


repo = CRUDExperiment(Experiment)
