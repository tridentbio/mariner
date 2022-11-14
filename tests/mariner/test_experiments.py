from mlflow.tracking.client import MlflowClient
import pytest
from mockito import patch
from sqlalchemy.orm.session import Session

from mariner import experiments as experiments_ctl
from mariner.entities import Experiment as ExperimentEntity
from mariner.schemas.experiment_schemas import (
    Experiment,
    ListExperimentsQuery,
    TrainingRequest,
)
from mariner.schemas.model_schemas import Model
from mariner.tasks import get_exp_manager
from tests.conftest import get_test_user
from tests.utils.utils import random_lower_string


@pytest.mark.asyncio
async def test_get_experiments(db: Session, some_model: Model, some_experiments):
    user = get_test_user(db)
    query = ListExperimentsQuery(model_id=some_model.id)
    experiments = experiments_ctl.get_experiments(db, user, query)
    assert len(experiments) == len(some_experiments) == 3


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

    # Await for tas
    await task

    db.commit()
    # Assertions over task outcome
    db_exp = db.query(ExperimentEntity).filter(ExperimentEntity.id == exp.id).first()
    db_exp = Experiment.from_orm(db_exp)
    assert db_exp.train_metrics
    assert db_exp.history
    assert "train_loss" in db_exp.train_metrics
    assert len(db_exp.history["train_loss"]) == request.epochs
    collected_regression_metrics = [
        "train_mse",
        "train_mae",
        "train_ev",
        "train_mape",
        "train_R2",
        "train_pearson",
        "train_spearman",
        "val_mse",
        "val_mae",
        "val_ev",
        "val_mape",
        "val_R2",
        "val_pearson",
        "val_spearman",
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
    )
    # Mock CustomLogger forward to raise an Exception
    import model_builder.model
    import mariner.train.run

    def _raise(_):
        raise Exception("bad bad model")

    # Patch remote ray training for local
    with patch(
        mariner.train.run.train_run_sync.remote,
        mariner.train.run.train_run,
    ):
        with patch(model_builder.model.CustomModel.forward, lambda x: _raise(x)):
            exp = await experiments_ctl.create_model_traning(db, user, request)
            assert exp.model_version_id == version.id
            assert exp.model_version.name == version.name
            task = get_exp_manager().get_task(exp.id)
            assert task
            # Await for tas
            with pytest.raises(Exception):
                await task

            db.commit()
            # Assertions over task outcome
            db_exp = (
                db.query(ExperimentEntity).filter(ExperimentEntity.id == exp.id).first()
            )
            db_exp = Experiment.from_orm(db_exp)
            assert db_exp.stack_trace
            assert len(db_exp.stack_trace) > 0
            assert db_exp.stage == "FAILED"
