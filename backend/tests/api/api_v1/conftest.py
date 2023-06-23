import io
from collections.abc import Generator
from typing import List

import pytest
from sqlalchemy.orm import Session

from mariner.core.aws import upload_s3_compressed
from mariner.entities import EventEntity
from mariner.entities import Experiment as ExperimentEntity
from mariner.entities import Model
from mariner.entities.dataset import Dataset
from mariner.schemas.dataset_schemas import DatasetCreateRepo
from mariner.schemas.experiment_schemas import Experiment
from mariner.schemas.token import TokenPayload
from mariner.stores.dataset_sql import dataset_store
from mariner.stores.experiment_sql import experiment_store
from tests.fixtures.events import get_test_events, teardown_events
from tests.fixtures.experiments import mock_experiment
from tests.fixtures.user import get_test_user
from tests.utils.utils import random_lower_string


@pytest.fixture(scope="module")
def mocked_experiment_payload(some_model: Model):
    experiment_name = random_lower_string()
    version = some_model.versions[-1]
    target_column = version.config.dataset.target_columns[0]
    return {
        "name": experiment_name,
        "modelVersionId": version.id,
        "framework": "torch",
        "config": {
            "optimizer": {
                "classPath": "torch.optim.Adam",
                "params": {
                    "lr": 0.05,
                },
            },
            "epochs": 1,
            "checkpointConfig": {
                "metricKey": f"val/mse/{target_column.name}",
                "mode": "min",
            },
            "earlyStoppingConfig": {
                "metricKey": f"val/mse/{target_column.name}",
                "mode": "min",
            },
        },
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


@pytest.fixture(scope="module")
def some_dataset_without_process(
    db: Session, normal_user_token_headers_payload: TokenPayload
) -> Dataset:
    """Fixture to create a dataset without process it
    This dataset has errors dataset key to download
    Dataset will be deleted after test
    """
    key, _ = upload_s3_compressed(
        io.BytesIO(open("tests/data/csv/bad_dataset.csv", "rb").read())
    )
    create_obj = DatasetCreateRepo.construct(
        bytes=0,
        rows=0,
        data_url=key,
        split_target="60-20-20",
        split_type="random",
        columns_metadata=[],
        name=random_lower_string(),
        description=random_lower_string(),
        created_by_id=normal_user_token_headers_payload.sub,
        errors={"dataset_error_key": key},
        columns=0,
        stats={},
    )
    dataset = dataset_store.create(db, create_obj)

    dataset_from_orm = dataset_store.get(db, dataset.id)
    yield dataset_from_orm

    db.query(Dataset).filter(Dataset.id == dataset.id).delete()
