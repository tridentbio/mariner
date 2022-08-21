import asyncio
from collections.abc import Generator

import pytest
from sqlalchemy.orm import Session

from app.features.experiments import controller as exp_ctl
from app.features.experiments.crud import repo as experiments_repo
from app.features.experiments.model import Experiment as ExperimentEntity
from app.features.experiments.schema import Experiment, TrainingRequest
from app.features.model.schema.model import Model
from app.tests.conftest import get_test_user
from app.tests.features.experiments.conftest import mock_experiment
from app.tests.utils.utils import random_lower_string


@pytest.fixture(scope="function")
def some_experiments(db, some_model: Model):
    db.query(ExperimentEntity).delete()
    db.commit()
    user = get_test_user(db)
    version = some_model.versions[-1]
    requests = [
        TrainingRequest(
            model_version_id=version.id,
            epochs=1,
            name=random_lower_string(),
            learning_rate=0.05,
        )
        for _ in range(3)
    ]
    exps = [exp_ctl.create_model_traning(db, user, request) for request in requests]
    exps = asyncio.get_event_loop().run_until_complete(asyncio.gather(*exps))
    yield exps
    # yield [Experiment.from_orm(exp) for exp in exps ]
    db.query(ExperimentEntity).delete()
    db.commit()


@pytest.fixture(scope="module")
def mocked_experiment_payload(some_model: Model):
    experiment_name = random_lower_string()
    version = some_model.versions[-1]
    return {
        "name": experiment_name,
        "learningRate": 0.05,
        "epochs": 1,
        "modelVersionId": version.id,
    }


@pytest.fixture(scope="function")
def some_experiment(
    db: Session, some_model: Model
) -> Generator[Experiment, None, None]:
    user = get_test_user(db)
    version = some_model.versions[-1]
    exp = experiments_repo.create(
        db, obj_in=mock_experiment(version, user.id, stage="started")
    )
    assert exp
    exp = Experiment.from_orm(exp)
    yield exp
    db.query(ExperimentEntity).filter(ExperimentEntity.id == exp.id).delete()
