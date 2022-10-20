from collections.abc import Generator
from typing import List

import pytest
from sqlalchemy.orm import Session

from app.features.events.event_model import EventEntity
from app.features.experiments.crud import repo as experiments_repo
from app.features.experiments.model import Experiment as ExperimentEntity
from app.features.experiments.schema import Experiment
from app.features.model.schema.model import Model
from app.tests.conftest import get_test_user
from app.tests.features.events.utils import get_test_events, teardown_events
from app.tests.features.experiments.conftest import mock_experiment
from app.tests.utils.utils import random_lower_string


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


@pytest.fixture(scope="function")
def some_events(
    db: Session, some_experiments: List[Experiment]
) -> Generator[List[EventEntity], None, None]:
    events = get_test_events(db, some_experiments)
    yield events
    teardown_events(db, events)
