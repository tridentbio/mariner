import asyncio
from typing import Coroutine, List

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal
from app.features.events.event_model import EventEntity
from app.features.experiments import controller as experiments_ctl
from app.features.experiments.schema import Experiment, TrainingRequest
from app.features.experiments.tasks import get_exp_manager
from app.features.model.schema.model import Model
from app.tests.conftest import get_test_user
from app.tests.utils.utils import random_lower_string


@pytest.mark.asyncio
async def test_get_notifications(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    some_model: Model,
    experiment_fixture,
):
    result = await experiment_fixture
    res = client.get(
        f"{settings.API_V1_STR}/events/report", headers=normal_user_token_headers
    )
    assert res.status_code == 200, "Request failed"
    body = res.json()
    expected_notification = {
        "source": "training:completed",
        "total": 1,
        "message": f'Training "{result.experiment_name}" completed',
    }
    got_notification = body[0]
    assert got_notification["source"] == expected_notification["source"]
    assert got_notification["total"] == expected_notification["total"]
    assert got_notification["message"] == expected_notification["message"]
    assert "events" in got_notification
    assert len(got_notification["events"]) == 1
    expected_url = f"{settings.WEBAPP_URL}/models/{some_model.id}#training"
    assert got_notification["events"][0]["url"] == expected_url


@pytest.mark.asyncio
async def test_post_read_notifications(
    db: Session,
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    experiments_fixture: Coroutine[List[Experiment], None, None],
    events_fixture: List[EventEntity],
):
    assert len(events_fixture) == 1
    print(experiments_fixture)
    result = await experiments_fixture
    res = client.get(
        f"{settings.API_V1_STR}/events/report",
        headers=normal_user_token_headers,
    )
    assert res.status_code == 200
    body = res.json()
    assert len(body) == 1
    assert body[0]["total"] == 1
    print("repor", body)

    res = client.post(
        f"{settings.API_V1_STR}/events/read",
        headers=normal_user_token_headers,
        json={"eventIds": [event.id for event in events_fixture]},
    )
    print("read", res.json())
    assert res.status_code == 200
    assert res.json() == {"total": 1}, "POST /events/read doesn't update any event"

    # assert all events are read
    res = client.get(
        f"{settings.API_V1_STR}/events/report",
        headers=normal_user_token_headers,
    )
    assert res.status_code == 200
    body = res.json()
    assert len(body) == 0


@pytest.fixture(scope="module")
async def experiment_fixture(db: Session, some_model: Model) -> Experiment:
    user = get_test_user(db)
    db.query(EventEntity).filter(EventEntity.user_id == user.id).delete()
    db.flush()
    version = some_model.versions[-1]
    request = TrainingRequest(
        model_version_id=version.id,
        epochs=1,
        name=random_lower_string(),
        learning_rate=0.05,
    )
    exp = await experiments_ctl.create_model_traning(db, user, request)
    task = get_exp_manager().get_task(exp.id)
    assert task, "Failed to get training async test"
    await task
    return exp


def teardown_module():
    # Cleans all events for the test user
    # ...
    db = SessionLocal()
    user = get_test_user(db)
    db.query(EventEntity).filter(EventEntity.user_id == user.id).delete()
    db.flush()
    db.close()


@pytest.fixture(scope="module")
async def experiments_fixture(db: Session, some_model: Model):
    user = get_test_user(db)
    db.query(EventEntity).filter(EventEntity.user_id == user.id).delete()
    db.flush()
    version = some_model.versions[-1]
    experiments = [
        await experiments_ctl.create_model_traning(
            db,
            user,
            TrainingRequest(
                name=random_lower_string(),
                epochs=1,
                learning_rate=0.1,
                model_version_id=version.id,
            ),
        )
    ]
    tasks = [await exp.task for exp in get_exp_manager().get_from_user(user.id)]
    return experiments


@pytest.fixture(scope="module")
def events_fixture(db: Session, experiments_fixture: List[Experiment]):
    user = get_test_user(db)
    return db.query(EventEntity).filter(EventEntity.user_id == user.id).all()
