"""
Experiment data layer defining ways to read and write to the experiments collection
"""
from typing import Any, Dict, List, Literal, Optional, Union

import pydantic
from sqlalchemy import Column
from sqlalchemy.orm import Query
from sqlalchemy.orm.session import Session

from mariner.entities import Experiment, ModelVersion
from mariner.schemas.api import OrderByQuery
from mariner.stores.base_sql import CRUDBase


class ExperimentCreateRepo(pydantic.BaseModel):
    """Payload to record an experiment in experiments collection."""

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
    """Payload to update an experiment in experiments collection."""

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
    """Data layer access on the experiments collection."""

    def _filter_by_model_id(self, query: Query, id: int) -> Query:
        return query.join(Experiment.model_version).filter(ModelVersion.model_id == id)

    def _filter_by_stages(self, query: Query, stages: List[str]) -> Query:
        return query.filter(Experiment.stage.in_(stages))

    def _by_creator(self, query: Query, creator_id: int) -> Query:
        query = query.filter(Experiment.created_by_id == creator_id)
        return query

    def _validate_order_by(self, query: OrderByQuery):
        ACCEPTED_ORDER_COLUMNS = [
            "createdAt",
        ]
        for clause in query.clauses:
            if clause.field not in ACCEPTED_ORDER_COLUMNS:
                raise ValueError(
                    f"{clause.field} is not accepted as orderBy."
                    f" Must be one of {', '.join(ACCEPTED_ORDER_COLUMNS)}"
                )

    def _add_order_by(self, sql_query: Query, query: OrderByQuery):
        for clause in query.clauses:
            col: Union[Column, None] = None
            if clause.field == "createdAt":
                col = Experiment.created_at

            if not col:
                continue

            if clause.order == "desc":
                sql_query = sql_query.order_by(col.desc())
            else:
                sql_query = sql_query.order_by(col)
        return sql_query

    def get_experiments_paginated(
        self,
        db: Session,
        model_id: Optional[int] = None,
        created_by_id: Optional[int] = None,
        stages: Union[List[str], None] = None,
        page: int = 1,
        order_by: Union[OrderByQuery, None] = None,
        per_page: int = 15,
    ) -> tuple[List[Experiment], int]:
        """Get's paginated view of experiments.

        Args:
            db: Connection to the database.
            model_id: Id of the model to filter the experiments.
            created_by_id: Id of the model creator to filter the models by.
            stages: Model stages to filter the models by.
            page: Number of the wanted page.
            order_by: Object describing wanted ordering of the data.
            per_page: Number of items per page.

        Returns:
            A tuple with a list of experiments and the total number of
        items satisfying the query without the pagination into account.
        """
        sql_query = db.query(Experiment)
        if model_id:
            sql_query = self._filter_by_model_id(sql_query, model_id)
        if created_by_id:
            sql_query = self._by_creator(sql_query, created_by_id)
        if stages:
            sql_query = self._filter_by_stages(query=sql_query, stages=stages)
        if order_by:
            self._validate_order_by(order_by)
            sql_query = self._add_order_by(sql_query, query=order_by)
        total = sql_query.count()
        sql_query = sql_query.limit(per_page)
        sql_query = sql_query.offset(per_page * page)
        result = sql_query.all()
        return result, total

    def get_experiments_metrics_for_model_version(
        self, db: Session, model_version_id: int, user: int
    ) -> List[Experiment]:
        """Get's all experiments for a given model version.

        Args:
            db: Connection to the database.
            model_version_id: Id of the model version to filter the experiments.
            user: Id of the user requesting the experiments.

        Returns:
            A list of experiments.
        """
        sql_query = db.query(Experiment)
        sql_query = sql_query.filter(Experiment.model_version_id == model_version_id)
        sql_query = self._by_creator(sql_query, user)
        return sql_query.all()


experiment_store = CRUDExperiment(Experiment)
