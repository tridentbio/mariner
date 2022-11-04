from asyncio.events import AbstractEventLoop
from typing import List

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from mariner import experiments as experiments_ctl
from mariner.core.config import settings
from mariner.db.session import SessionLocal
from mariner.entities import EventEntity
from mariner.schemas.experiment_schemas import Experiment, TrainingRequest
from mariner.schemas.model_schemas import Model
from mariner.tasks import get_exp_manager
from tests.conftest import get_test_user
from tests.utils.utils import random_lower_string


@pytest.fixture(scope="module")
async def experiments_fixture(db: Session, some_model: Model):
    user = get_test_user(db)
    db.query(EventEntity).delete()
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
    tasks = get_exp_manager().get_from_user(user.id)
    for task in tasks:
        await task.task
    return experiments


@pytest.fixture(scope="module")
async def events_fixture(db: Session, experiments_fixture):
    user = get_test_user(db)
    events_ents = db.query(EventEntity).filter(EventEntity.user_id == user.id).all()
    return events_ents


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
    events_fixture: List[EventEntity],
    event_loop: AbstractEventLoop,
):
    res = client.get(
        f"{settings.API_V1_STR}/events/report",
        headers=normal_user_token_headers,
    )
    events_fixture = await events_fixture
    assert res.status_code == 200
    body = res.json()
    assert len(body) == 1
    assert body[0]["total"] == 1

    res = client.post(
        f"{settings.API_V1_STR}/events/read",
        headers=normal_user_token_headers,
        json={"eventIds": [event.id for event in events_fixture]},
    )
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
