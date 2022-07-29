from typing import Generator, List, Literal, Optional

import pytest
from sqlalchemy.orm.session import Session

from app.features.experiments.crud import ExperimentCreateRepo
from app.features.experiments.crud import repo as experiments_repo
from app.features.experiments.schema import Experiment
from app.features.model.schema.model import Model, ModelVersion
from app.tests.conftest import get_test_user
from app.tests.utils.utils import random_lower_string


def mock_experiment(
    some_model: Model,
    version: ModelVersion,
    user_id: int,
    stage: Optional[Literal["started", "success"]] = None,
):
    create_obj = ExperimentCreateRepo(
        experiment_id=random_lower_string(),
        created_by_id=user_id,
        model_name=some_model.name,
        model_version_name=version.model_version,
    )
    if stage == "started":
        pass  # create_obj is ready
    elif stage == "success":
        create_obj.history = {
            "train_loss": [300.3, 210.9, 160.8, 130.3, 80.4, 50.1, 20.0]
        }
        create_obj.train_metrics = {"train_loss": 200.3}
        create_obj.stage = "SUCCESS"
    else:
        raise NotImplementedError()
    return create_obj


@pytest.fixture(scope="function")
def some_experiment(
    db: Session, some_model: Model
) -> Generator[Experiment, None, None]:
    user = get_test_user(db)
    version = some_model.versions[-1]
    exp = experiments_repo.create(
        db, obj_in=mock_experiment(some_model, version, user.id, stage="started")
    )
    assert exp
    exp = Experiment.from_orm(exp)
    yield exp
    db.query(Experiment).filter(Experiment.experiment_id == exp.experiment_id).delete()


@pytest.fixture(scope="function")
def some_cmoplete_experiment(
    db: Session, some_model: Model
) -> Generator[Experiment, None, None]:
    user = get_test_user(db)
    version = some_model.versions[-1]
    exp = experiments_repo.create(
        db, obj_in=mock_experiment(some_model, version, user.id, stage="success")
    )
    assert exp
    exp = Experiment.from_orm(exp)
    yield exp
    db.query(Experiment).filter(Experiment.experiment_id == exp.experiment_id).delete()
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
            db, obj_in=mock_experiment(some_model, version, user.id, stage="started")
        )
        for _ in range(0, 3)
    ]
    exps = [Experiment.from_orm(exp) for exp in exps]
    yield exps
    db.query(Experiment).delete()
    db.commit()
