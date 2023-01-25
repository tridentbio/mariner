import pytest
from mlflow.tracking.client import MlflowClient
from mockito import patch
from sqlalchemy.orm.session import Session

from mariner import experiments as experiments_ctl
from mariner.entities import Experiment as ExperimentEntity
from mariner.schemas.api import OrderByClause, OrderByQuery
from mariner.schemas.experiment_schemas import (
    EarlyStoppingConfig,
    Experiment,
    ListExperimentsQuery,
    MonitoringConfig,
    TrainingRequest,
)
from mariner.schemas.model_schemas import Model
from mariner.tasks import get_exp_manager
from tests.fixtures.user import get_test_user
from tests.utils.utils import random_lower_string


@pytest.mark.asyncio
async def test_get_experiments_ordered_by_createdAt_asceding(
    db: Session, some_model: Model, some_experiments
):
    user = get_test_user(db)
    query = ListExperimentsQuery(
        model_id=some_model.id,
        stage=None,
        page=0,
        per_page=15,
        model_version_ids=None,
        order_by=OrderByQuery(clauses=[OrderByClause(field="createdAt", order="asc")]),
    )
    experiments, total = experiments_ctl.get_experiments(db, user, query)
    assert (
        len(experiments) == len(some_experiments) == total == 3
    ), "query gets all experiments"
    for i in range(len(experiments[:-1])):
        current, next = experiments[i], experiments[i + 1]
        assert (
            next.created_at > current.created_at
        ), "createdAt asceding order is not respected"


@pytest.mark.asyncio
async def test_get_experiments_ordered_by_createdAt_desceding(
    db: Session, some_model: Model, some_experiments
):
    user = get_test_user(db)
    query = ListExperimentsQuery(
        model_id=some_model.id,
        stage=None,
        page=0,
        per_page=15,
        model_version_ids=None,
        order_by=OrderByQuery(clauses=[OrderByClause(field="createdAt", order="desc")]),
    )
    experiments, total = experiments_ctl.get_experiments(db, user, query)
    assert (
        len(experiments) == len(some_experiments) == total == 3
    ), "query gets all experiments"
    for i in range(len(experiments[:-1])):
        current, next = experiments[i], experiments[i + 1]
        assert (
            next.created_at < current.created_at
        ), "createdAt descending order is not respected"


@pytest.mark.asyncio
@pytest.mark.long
async def test_create_model_training(db: Session, some_model: Model):
    user = get_test_user(db)
    version = some_model.versions[-1]
    request = TrainingRequest(
        model_version_id=version.id,
        epochs=1,
        name=random_lower_string(),
        learning_rate=0.05,
        checkpoint_config=MonitoringConfig(
            mode="min",
            metric_key="val_mse",
        ),
        early_stopping_config=EarlyStoppingConfig(metric_key="val_mse", mode="min"),
    )
    exp = await experiments_ctl.create_model_traning(db, user, request)
    assert exp.model_version_id == version.id
    assert exp.model_version.name == version.name
    task = get_exp_manager().get_task(exp.id)
    assert task

    # Assertions before task completion
    db_exp = Experiment.from_orm(
        db.query(ExperimentEntity).filter(ExperimentEntity.id == exp.id).first()
    )
    assert db_exp.model_version_id == exp.model_version_id
    assert db_exp.created_by_id == user.id
    assert db_exp.epochs == request.epochs

    # Await for task
    await task

    db.commit()
    # Assertions over task outcome
    db_exp = db.query(ExperimentEntity).filter(ExperimentEntity.id == exp.id).first()
    db_exp = Experiment.from_orm(db_exp)
    assert db_exp.train_metrics
    assert db_exp.history
    assert db_exp.stage == "SUCCESS"
    assert "train_loss" in db_exp.train_metrics
    assert len(db_exp.history["train_loss"]) == request.epochs
    collected_regression_metrics = [
        "train_mse",
        "train_mae",
        "train_ev",
        "train_mape",
        "train_R2",
        "train_pearson",
    ]
    for metric in collected_regression_metrics:
        assert len(db_exp.history[metric]) == request.epochs
    client = MlflowClient()
    experiment = client.get_experiment(db_exp.mlflow_id)
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    for metric_key in collected_regression_metrics:
        metric = client.get_metric_history(run_id=run.info.run_id, key=metric_key)
        assert metric
        assert len(metric) == request.epochs


@pytest.mark.asyncio
@pytest.mark.skip("Cant patch class that's called from ray worker.")
async def test_experiment_has_stacktrace_when_training_fails(
    db: Session, some_model: Model
):
    user = get_test_user(db)
    version = some_model.versions[-1]
    request = TrainingRequest(
        model_version_id=version.id,
        epochs=1,
        name=random_lower_string(),
        learning_rate=0.05,
        checkpoint_config=MonitoringConfig(
            mode="min",
            metric_key="val_mse",
        ),
        early_stopping_config=EarlyStoppingConfig(metric_key="val_mse", mode="min"),
    )
    # Mock CustomLogger forward to raise an Exception
    import model_builder.model

    def _raise(_):
        raise Exception("bad bad model")

    # Patch remote ray training for local
    with patch(model_builder.model.CustomModel.forward, lambda x: _raise(x)):
        exp = await experiments_ctl.create_model_traning(db, user, request)
        assert exp.model_version_id == version.id
        assert exp.model_version.name == version.name
        task = get_exp_manager().get_task(exp.id)
        assert task
        # Await for tas
        with pytest.raises(Exception):
            result = await task
            print(result)

        db.commit()
        # Assertions over task outcome
        db_exp = (
            db.query(ExperimentEntity).filter(ExperimentEntity.id == exp.id).first()
        )
        db_exp = Experiment.from_orm(db_exp)
        assert db_exp.stack_trace
        assert len(db_exp.stack_trace) > 0
        assert db_exp.stage == "ERROR"
