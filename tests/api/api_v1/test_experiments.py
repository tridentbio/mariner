from datetime import datetime
from urllib.parse import urlencode

import pytest
from fastapi import status
from sqlalchemy.orm import Session
from starlette.status import HTTP_200_OK
from starlette.testclient import TestClient

from mariner.core.config import settings
from mariner.schemas.experiment_schemas import Experiment
from tests.fixtures.user import get_test_user


@pytest.mark.long
def test_post_experiments(
    client: TestClient, mocked_experiment_payload: dict, normal_user_token_headers
):
    res = client.post(
        f"{settings.API_V1_STR}/experiments/",
        json=mocked_experiment_payload,
        headers=normal_user_token_headers,
    )
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
    body = res.json()
    exps, total = body["data"], body["total"]
    assert len(exps) == len(some_experiments) == total == 3, "gets all experiments"


def test_get_experiments_ordered_by_createdAt_desc_url_encoded(
    client: TestClient, some_model, some_experiments, normal_user_token_headers
):
    params = {"modelId": some_model.id, "orderBy": "-createdAt"}
    querystring = urlencode(params)

    res = client.get(
        f"{settings.API_V1_STR}/experiments/?{querystring}",
        params=params,
        headers=normal_user_token_headers,
    )
    assert res.status_code == HTTP_200_OK
    body = res.json()
    exps, total = body["data"], body["total"]
    assert len(exps) == len(some_experiments) == total == 3, "gets all experiments"
    for i in range(len(exps[:-1])):
        current, next = exps[i], exps[i + 1]
        assert datetime.fromisoformat(next["createdAt"]) < datetime.fromisoformat(
            current["createdAt"]
        ), "createdAt descending order is not respected"


def test_get_experiments_ordered_by_createdAt_desc(
    client: TestClient, some_model, some_experiments, normal_user_token_headers
):
    params = {"modelId": some_model.id, "orderBy": "-createdAt"}

    res = client.get(
        f"{settings.API_V1_STR}/experiments/",
        params=params,
        headers=normal_user_token_headers,
    )
    assert res.status_code == HTTP_200_OK
    body = res.json()
    exps, total = body["data"], body["total"]
    assert len(exps) == len(some_experiments) == total == 3, "gets all experiments"
    for i in range(len(exps[:-1])):
        current, next = exps[i], exps[i + 1]
        assert datetime.fromisoformat(next["createdAt"]) < datetime.fromisoformat(
            current["createdAt"]
        ), "createdAt descending order is not respected"


def test_get_experiments_ordered_by_createdAt_asc_url_encoded(
    client: TestClient, some_model, some_experiments, normal_user_token_headers
):
    params = {"modelId": some_model.id, "orderBy": "+createdAt"}
    querystring = urlencode(params)

    res = client.get(
        f"{settings.API_V1_STR}/experiments/?{querystring}",
        headers=normal_user_token_headers,
    )
    assert res.status_code == HTTP_200_OK
    body = res.json()
    exps, total = body["data"], body["total"]
    assert len(exps) == len(some_experiments) == total == 3, "gets all experiments"
    for i in range(len(exps[:-1])):
        current, next = exps[i], exps[i + 1]
        assert datetime.fromisoformat(next["createdAt"]) > datetime.fromisoformat(
            current["createdAt"]
        ), "createdAt ascending order is not respected"


# FAILING
def test_get_experiments_by_stage(
    client: TestClient, some_model, some_experiments, normal_user_token_headers
):
    params = {
        "modelId": some_model.id,
        "page": 0,
        "perPage": 15,
        "stage": ["SUCCESS"],
    }
    res = client.get(
        f"{settings.API_V1_STR}/experiments",
        params=params,
        headers=normal_user_token_headers,
    )
    body = res.json()
    assert res.status_code == HTTP_200_OK, "Request failed with body: %r" % body
    exps, total = body["data"], body["total"]
    for exp in exps:
        assert exp["stage"] == "SUCCESS", "experiment out of stage filter"

    assert (
        len(exps) == total == 2
    ), "request failed to get the 2 out 3 successfull experiments"


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
