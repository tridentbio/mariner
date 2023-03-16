from typing import Dict, Generator, List

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from jose import jwt
from sqlalchemy.orm import Session

from api.fastapi_app import app
from mariner import experiments as experiments_ctl
from mariner.core import security
from mariner.core.config import settings
from mariner.db.session import SessionLocal
from mariner.entities import Dataset, EventEntity
from mariner.entities import Experiment as ExperimentEntity
from mariner.entities import Model as ModelEntity
from mariner.entities import User
from mariner.entities.event import EventReadEntity
from mariner.schemas.experiment_schemas import (
    EarlyStoppingConfig,
    Experiment,
    MonitoringConfig,
    TrainingRequest,
)
from mariner.schemas.model_schemas import Model
from mariner.schemas.token import TokenPayload
from mariner.stores import user_sql
from mariner.stores.experiment_sql import experiment_store
from mariner.tasks import get_exp_manager
from model_builder.optimizers import AdamOptimizer
from tests.fixtures.dataset import (
    setup_create_dataset,
    setup_create_dataset_db,
    teardown_create_dataset,
)
from tests.fixtures.events import get_test_events, teardown_events
from tests.fixtures.experiments import (
    mock_experiment,
    setup_experiments,
    teardown_experiments,
)
from tests.fixtures.model import (
    setup_create_model,
    setup_create_model_db,
    teardown_create_model,
)
from tests.fixtures.user import get_test_user
from tests.utils.user import authentication_token_from_email, create_random_user
from tests.utils.utils import random_lower_string


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
def normal_user_token_headers_payload(
    normal_user_token_headers: Dict[str, str]
) -> Dict[str, str]:
    """Get the payload from the token"""
    token = normal_user_token_headers["Authorization"].split(" ")[1]
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[security.ALGORITHM])
    return TokenPayload(**payload)


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
    ds = setup_create_dataset_db(db)
    assert ds is not None
    yield ds
    teardown_create_dataset(db, ds)


@pytest.fixture(scope="module")
def some_bio_dataset(
    db: Session, client: TestClient, normal_user_token_headers: Dict[str, str]
):
    ds = setup_create_dataset(
        db, client, normal_user_token_headers, file="tests/data/csv/chimpanzee.csv"
    )
    assert ds is not None
    yield ds
    teardown_create_dataset(db, ds)


# MODEL GLOBAL FIXTURES
@pytest.fixture(scope="module")
def some_model(
    db: Session,
    some_dataset: Dataset,
):
    """Model fixture

    Creates a fixture model for unit testing. Fails
    database service is down

    Args:
        db: database connection
        client: fastapi http client
        normal_user_token_headers: authenticated headers
        some_dataset: dataset to be used on model
    """
    model = setup_create_model_db(
        db=db,
        dataset=some_dataset,
        model_type="regressor",
    )
    yield model
    teardown_create_model(db, model, skip_mlflow=True)


@pytest.fixture(scope="module")
def some_model_integration(
    db: Session,
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    some_dataset: Dataset,
):
    """Model fixture

    Creates a fixture model using running services. Fails
    if mlflow, ray or database services are down

    Args:
        db: database connection
        client: fastapi http client
        normal_user_token_headers: authenticated headers
        some_dataset: dataset to be used on model
    """
    model = setup_create_model(
        client,
        normal_user_token_headers,
        dataset_name=some_dataset.name,
        model_type="regressor",
    )
    yield model
    teardown_create_model(db, model)


@pytest_asyncio.fixture(scope="function")
async def some_trained_model(
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
    version = model.versions[-1]
    target_column = version.config.dataset.target_columns[0]
    request = TrainingRequest(
        model_version_id=version.id,
        epochs=1,
        name=random_lower_string(),
        optimizer=AdamOptimizer(),
        checkpoint_config=MonitoringConfig(
            mode="min",
            metric_key=f"val_mse_{target_column.name}",
        ),
        early_stopping_config=EarlyStoppingConfig(
            metric_key=f"val_mse_{target_column.name}", mode="min"
        ),
    )
    user = user_sql.user_store.get(db, model.created_by_id)
    assert user
    exp = await experiments_ctl.create_model_traning(db, user, request)
    task = get_exp_manager().get_task(exp.id)
    assert task
    await task
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
def some_successfull_experiment(
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
    exps = setup_experiments(db, some_model, num_experiments=3)
    assert len(exps) == 3, "failed in setup of some_experiments fixture"
    yield exps
    teardown_experiments(db, exps)
