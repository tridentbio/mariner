from collections.abc import Generator
from typing import List

import pytest
from sqlalchemy.orm import Session

from mariner.entities import EventEntity
from mariner.entities import Experiment as ExperimentEntity
from mariner.entities import Model
from mariner.schemas.experiment_schemas import Experiment
from mariner.stores.experiment_sql import experiment_store
from tests.fixtures.events import get_test_events, teardown_events
from tests.fixtures.experiments import mock_experiment
from tests.fixtures.user import get_test_user
from tests.utils.utils import random_lower_string


@pytest.fixture(scope="module")
def mocked_experiment_payload(some_model: Model):
    experiment_name = random_lower_string()
    version = some_model.versions[-1]
    return {
        "name": experiment_name,
        "optimizer": {
            "classPath": "torch.optim.Adam",
            "params": {
                "lr": 0.05,
            },
        },
        "epochs": 1,
        "modelVersionId": version.id,
        "checkpointConfig": {"metricKey": "val_mse", "mode": "min"},
        "earlyStoppingConfig": {"metricKey": "val_mse", "mode": "min"},
    }


@pytest.fixture(scope="function")
def some_experiment(
    db: Session, some_model: Model
) -> Generator[Experiment, None, None]:
    user = get_test_user(db)
    version = some_model.versions[-1]
    exp = experiment_store.create(
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
