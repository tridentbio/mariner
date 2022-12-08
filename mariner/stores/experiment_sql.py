from typing import Any, Dict, List, Literal, Optional, Union

import pydantic
from sqlalchemy.orm import Query
from sqlalchemy.orm.session import Session

from mariner.entities import Experiment, ModelVersion
from mariner.schemas.experiment_schemas import ListExperimentsQuery
from mariner.stores.base_sql import CRUDBase


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
        Literal["NOT RUNNING", "STARTED", "RUNNING", "ERROR", "SUCCESS"]
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

    def get_by_model_id(self, db: Session, id: int) -> List[Experiment]:
        query = db.query(Experiment)
        query = self.filter_by_model_id(query, id)
        return query.all()

    def get_many(self, db: Session, q: ListExperimentsQuery) -> List[Experiment]:
        query = db.query(Experiment)
        if q.model_id:
            query = self.filter_by_model_id(query, q.model_id)
        if q.stage:
            query = self.filter_by_stages(query, q.stage)
        return query.all()

    def _by_creator(self, query: Query, creator_id: int) -> Query:
        query = query.filter(Experiment.created_by_id == creator_id)
        return query

    def get_experiments_paginated(
        self,
        db: Session,
        model_id: Optional[int] = None,
        created_by_id: Optional[int] = None,
        stages: Union[List[str], None] = None,
        page: int = 1,
        per_page: int = 15,
    ) -> tuple[List[Experiment], int]:
        sql_query = db.query(Experiment)
        if model_id:
            sql_query = self.filter_by_model_id(sql_query, model_id)
        if created_by_id:
            sql_query = self._by_creator(sql_query, created_by_id)
        if stages:
            sql_query = self.filter_by_stages(query=sql_query, stages=stages)
        total = sql_query.count()
        sql_query = sql_query.limit(per_page)
        sql_query = sql_query.offset(per_page * page)
        result = sql_query.all()
        return result, total

    def get(self, db: Session, experiment_id: int) -> Experiment:
        return db.query(Experiment).filter(Experiment.id == experiment_id).first()


experiment_store = CRUDExperiment(Experiment)
