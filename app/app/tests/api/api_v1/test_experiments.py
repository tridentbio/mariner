from fastapi import status
from sqlalchemy.orm import Session
from starlette.status import HTTP_200_OK
from starlette.testclient import TestClient

from app.core.config import settings
from app.features.experiments.schema import Experiment
from app.tests.conftest import get_test_user


def test_post_experiments(
    client: TestClient, mocked_experiment_payload: dict, normal_user_token_headers
):
    res = client.post(
        f"{settings.API_V1_STR}/experiments/",
        json=mocked_experiment_payload,
        headers=normal_user_token_headers,
    )
    print(res.json())
    assert res.status_code == HTTP_200_OK


def test_get_experiments(
    client: TestClient, some_model, some_experiments, normal_user_token_headers
):
    params = {"modelId": some_model.id}

    res = client.get(
        f"{settings.API_V1_STR}/experiments/",
        params=params,
        headers=normal_user_token_headers,
    )
    assert res.status_code == HTTP_200_OK
    exps = res.json()
    assert len(exps) > 1


def test_post_update_metrics_unauthorized(
    client: TestClient, db: Session, some_experiment: Experiment
):
    user_id = get_test_user(db).id
    metrics_update = {
        "type": "epochMetrics",
        "data": {"metrics": [], "epoch": 1},
        "experimentId": some_experiment.id,
        "experimentName": "",
        "userId": user_id,
    }
    res = client.post(
        f"{settings.API_V1_STR}/experiments/epoch_metrics",
        json=metrics_update,
    )
    assert res.status_code == status.HTTP_403_FORBIDDEN


def test_post_update_metrics_sucess(
    client: TestClient,
    db: Session,
    some_experiment: Experiment,
):
    user_id = get_test_user(db).id
    metrics_update = {
        "type": "epochMetrics",
        "data": {"metrics": [], "epoch": 1},
        "experimentId": some_experiment.id,
        "experimentName": "",
        "userId": user_id,
    }
    res = client.post(
        f"{settings.API_V1_STR}/experiments/epoch_metrics",
        json=metrics_update,
        headers={"Authorization": f"Bearer {settings.APPLICATION_SECRET}"},
    )
    assert res.status_code == status.HTTP_200_OK
