import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import settings
from app.features.experiments import controller as experiments_ctl
from app.features.experiments.schema import Experiment, TrainingRequest
from app.features.experiments.tasks import get_exp_manager
from app.features.model.schema.model import Model
from app.tests.conftest import get_test_user
from app.tests.utils.utils import random_lower_string


@pytest.fixture(scope="module")
@pytest.mark.asyncio
async def experiment_fixture(db: Session, some_model: Model) -> Experiment:
    user = get_test_user(db)
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


def setup_module():
    # Cleans all events for the test user
    # ...
    pass


def test_get_notifications(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    experiment_fixture: Experiment,
):
    res = client.get(
        f"{settings.API_V1_STR}/notifications/report", headers=normal_user_token_headers
    )
    assert res.status_code == 200, "Request failed"
    body = res.json()
    expected_notification = {
        "source": "training:created",
        "total": 1,
        "message": f'Training "{experiment_fixture.experiment_name}" completed',
    }
    got_notification = body[0]
    assert set(expected_notification.items()).issubset(got_notification)
    assert "eventIds" in got_notification
    assert len(got_notification["eventIds"]) == 1


def test_post_read_notifications(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    experiment_fixture: Experiment,
):
    events = ...  # get all notifications
    res = client.post(
        f"{settings.API_V1_STR}/notifications/read",
        headers=normal_user_token_headers,
        json={"eventIds": [event.id for event in events]},
    )
    assert res.status_code == 200
    # assert all events are read
