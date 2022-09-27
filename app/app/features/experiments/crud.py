from typing import Any, Dict, List, Literal, Optional

import pydantic
from sqlalchemy.orm import Query
from sqlalchemy.orm.session import Session

from app.crud.base import CRUDBase
from app.features.experiments.model import Experiment
from app.features.experiments.schema import ListExperimentsQuery
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
    def filter_by_model_id(self, query: Query, id: int) -> Query:
        return query.join(Experiment.model_version).filter(ModelVersion.model_id == id)

    def filter_by_stages(self, query: Query, stages: List[str]) -> Query:
        return query.filter(Experiment.stage.in_(stages))

    def get_by_model_id(self, db: Session, id: int):
        query = db.query(Experiment)
        query = self.filter_by_model_id(query, id)
        return query.all()

    def get_many(self, db: Session, q: ListExperimentsQuery):
        query = db.query(Experiment)
        if q.model_id:
            query = self.filter_by_model_id(query, q.model_id)
        if q.stage:
            query = self.filter_by_stages(query, q.stage)
        return query.all()

    def get(self, db: Session, experiment_id: int):
        return db.query(Experiment).filter(Experiment.id == experiment_id).first()


repo = CRUDExperiment(Experiment)
