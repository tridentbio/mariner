from typing import Generator, List, Literal, Optional

import pytest
from sqlalchemy.orm.session import Session

from app.features.experiments.crud import ExperimentCreateRepo
from app.features.experiments.crud import repo as experiments_repo
from app.features.experiments.model import Experiment as ExperimentEntity
from app.features.experiments.schema import Experiment
from app.features.model.schema.model import Model, ModelVersion
from app.tests.conftest import get_test_user
from app.tests.utils.utils import random_lower_string


def mock_experiment(
    version: ModelVersion,
    user_id: int,
    stage: Optional[Literal["started", "success"]] = None,
):
    create_obj = ExperimentCreateRepo(
        epochs=1,
        mlflow_id=random_lower_string(),
        created_by_id=user_id,
        model_version_id=version.id,
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
