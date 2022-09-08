from typing import Generator, List, Literal, Optional

import pytest
from sqlalchemy.orm.session import Session

from app.features.experiments.crud import ExperimentCreateRepo
from app.features.experiments.crud import repo as experiments_repo
from app.features.experiments.model import Experiment as ExperimentEntity
from app.features.experiments.schema import Experiment
from app.features.model.schema.model import Model, ModelVersion
from app.tests.conftest import get_test_user, mock_experiment
from app.tests.utils.utils import random_lower_string


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
def some_cmoplete_experiment(
    db: Session, some_model: Model
) -> Generator[Experiment, None, None]:
    user = get_test_user(db)
    version = some_model.versions[-1]
    exp = experiments_repo.create(
        db, obj_in=mock_experiment(version, user.id, stage="success")
    )
    assert exp
    exp = Experiment.from_orm(exp)
    yield exp
    db.query(ExperimentEntity).filter(ExperimentEntity.id == exp.id).delete()
    db.commit()


@pytest.fixture(scope="function")
def some_experiments(
    db: Session, some_model: Model
) -> Generator[List[Experiment], None, None]:
    user = get_test_user(db)
    version = some_model.versions[-1]

    user = get_test_user(db)
    version = some_model.versions[-1]
    exps = [
        experiments_repo.create(
            db, obj_in=mock_experiment(version, user.id, stage="started")
        )
        for _ in range(0, 3)
    ]
    exps = [Experiment.from_orm(exp) for exp in exps]
    yield exps
    db.query(ExperimentEntity).filter(
        ExperimentEntity.id.in_([exp.id for exp in exps])
    ).delete()
    db.flush()
