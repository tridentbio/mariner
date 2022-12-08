from typing import Dict, Generator, List

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from api.fastapi_app import app
from mariner.core.config import settings
from mariner.db.session import SessionLocal
from mariner.entities import Dataset, EventEntity
from mariner.entities import Experiment as ExperimentEntity
from mariner.entities import Model as ModelEntity
from mariner.entities import User
from mariner.entities.event import EventReadEntity
from mariner.schemas.experiment_schemas import Experiment
from mariner.schemas.model_schemas import Model
from mariner.stores.experiment_sql import experiment_store
from tests.fixtures.dataset import setup_create_dataset, teardown_create_dataset
from tests.fixtures.events import get_test_events, teardown_events
from tests.fixtures.experiments import mock_experiment
from tests.fixtures.model import setup_create_model, teardown_create_model
from tests.fixtures.user import get_test_user
from tests.utils.user import authentication_token_from_email, create_random_user


@pytest.fixture(scope="session")
def db() -> Generator:
    sesion = SessionLocal()
    yield sesion


@pytest.fixture(scope="module")
def client() -> Generator:
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def normal_user_token_headers(client: TestClient, db: Session) -> Dict[str, str]:
    return authentication_token_from_email(
        client=client, email=settings.EMAIL_TEST_USER, db=db
    )


@pytest.fixture(scope="module")
def user_fixture(db: Session):
    return create_random_user(db)


@pytest.fixture(scope="module")
def randomuser_token_headers(
    client: TestClient, db: Session, user_fixture: User
) -> Dict[str, str]:
    return authentication_token_from_email(
        client=client, email=user_fixture.email, db=db
    )


@pytest.fixture(scope="module")
def some_dataset(
    db: Session, client: TestClient, normal_user_token_headers: Dict[str, str]
):
    ds = setup_create_dataset(db, client, normal_user_token_headers)
    assert ds is not None
    yield ds
    teardown_create_dataset(db, ds)


# MODEL GLOBAL FIXTURES
@pytest.fixture(scope="module")
def some_model(
    db: Session,
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    some_dataset: Dataset,
):
    model = setup_create_model(
        client,
        normal_user_token_headers,
        dataset_name=some_dataset.name,
        model_type="regressor",
    )
    yield model
    teardown_create_model(db, model)


@pytest.fixture
def model(
    db: Session,
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    some_dataset: Dataset,
):
    db.commit()
    model = setup_create_model(
        client,
        normal_user_token_headers,
        dataset_name=some_dataset.name,
    )
    yield model
    dbobj = db.query(ModelEntity).filter(ModelEntity.id == model.id).first()
    if dbobj:
        db.delete(dbobj)
        db.flush()


@pytest.fixture(scope="function")
def classifier_model(
    db: Session,
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    some_dataset: Dataset,
):
    db.commit()
    model = setup_create_model(
        client,
        normal_user_token_headers,
        dataset_name=some_dataset.name,
        model_type="classifier",
    )
    yield model
    dbobj = db.query(ModelEntity).filter(ModelEntity.id == model.id).first()
    if dbobj:
        db.delete(dbobj)
        db.flush()


@pytest.fixture(scope="module")
def events_fixture(db: Session, some_experiments: List[Experiment]):
    db.query(EventReadEntity).delete()
    db.query(EventEntity).delete()
    db.flush()
    assert len(some_experiments) == 3
    events = get_test_events(db, some_experiments)
    assert len(events) == 3
    yield events
    teardown_events(db, events)


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
def some_cmoplete_experiment(
    db: Session, some_model: Model
) -> Generator[Experiment, None, None]:
    user = get_test_user(db)
    version = some_model.versions[-1]
    exp = experiment_store.create(
        db, obj_in=mock_experiment(version, user.id, stage="success")
    )
    assert exp
    exp = Experiment.from_orm(exp)
    yield exp
    db.query(ExperimentEntity).filter(ExperimentEntity.id == exp.id).delete()
    db.commit()


@pytest.fixture(scope="module")
def some_experiments(
    db: Session, some_model: Model
) -> Generator[List[Experiment], None, None]:
    user = get_test_user(db)
    version = some_model.versions[-1]
    # creates 1 started experiment and 2 successful
    exps = [
        experiment_store.create(
            db,
            obj_in=mock_experiment(
                version, user.id, stage="started" if i % 2 == 1 else "success"
            ),
        )
        for i in range(0, 3)
    ]
    exps = [Experiment.from_orm(exp) for exp in exps]
    assert len(exps) == 3, "failed in setup of some_experiments fixture"
    yield exps
    db.query(ExperimentEntity).filter(
        ExperimentEntity.id.in_([exp.id for exp in exps])
    ).delete()
    db.flush()
