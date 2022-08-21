import time

import pytest
from sqlalchemy.orm.session import Session

from app.features.experiments import controller as experiments_ctl
from app.features.experiments.model import Experiment as ExperimentEntity
from app.features.experiments.schema import (
    Experiment,
    ListExperimentsQuery,
    TrainingRequest,
)
from app.features.experiments.tasks import get_exp_manager
from app.features.model.schema.model import Model
from app.tests.conftest import get_test_user
from app.tests.utils.utils import random_lower_string


@pytest.mark.asyncio
async def test_get_experiments(db: Session, some_model: Model, some_experiments):
    user = get_test_user(db)
    query = ListExperimentsQuery(model_id=some_model.id)
    experiments = experiments_ctl.get_experiments(db, user, query)
    assert len(experiments) == len(some_experiments) == 3


@pytest.mark.asyncio
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
    task = get_exp_manager().get_task(exp.mlflow_id)
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
    time.sleep(5)

    # Assertions over task outcome
    db_exp = Experiment.from_orm(
        db.query(ExperimentEntity).filter(ExperimentEntity.id == exp.id).first()
    )
    assert db_exp.train_metrics
    assert db_exp.history
    assert "train_loss" in db_exp.train_metrics
    assert len(db_exp.history["train_loss"]) == request.epochs
