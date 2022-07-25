import pytest
from sqlalchemy.orm.session import Session

from app.features.experiments import controller as experiments_ctl
from app.features.experiments.model import Experiment
from app.features.experiments.schema import (
    ListExperimentsQuery,
    TrainingRequest,
)
from app.features.model.schema.model import Model
from app.tests.conftest import get_test_user
from app.tests.utils.utils import random_lower_string


@pytest.mark.asyncio
async def test_get_experiments(db: Session, some_model: Model, some_experiments):
    user = get_test_user(db)
    query = ListExperimentsQuery(model_name=some_model.name)
    experiments = experiments_ctl.get_experiments(db, user, query)
    assert len(experiments) == len(some_experiments) == 3


@pytest.mark.asyncio
async def test_create_model_training(db: Session, some_model: Model):
    user = get_test_user(db)
    version = some_model.versions[-1]
    request = TrainingRequest(
        model_name=some_model.name,
        model_version=version.model_version,
        epochs=1,
        experiment_name=random_lower_string(),
        learning_rate=0.05,
    )
    exp = await experiments_ctl.create_model_traning(db, user, request)
    assert exp.model_name == some_model.name

    assert exp.model_version.model_version == version.model_version
    db_exp = (
        db.query(Experiment)
        .filter(Experiment.experiment_id == exp.experiment_id)
        .first()
    )
    assert db_exp.experiment_id == exp.experiment_id
    assert db_exp.model_name == exp.model_name
