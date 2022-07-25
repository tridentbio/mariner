from starlette.status import HTTP_200_OK
from starlette.testclient import TestClient

from app.core.config import settings


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
    params = {"modelName": some_model.name}
    res = client.get(
        f"{settings.API_V1_STR}/experiments/",
        params=params,
        headers=normal_user_token_headers,
    )
    assert res.status_code == HTTP_200_OK
    exps = res.json()
    assert len(exps) == len(some_experiments)
