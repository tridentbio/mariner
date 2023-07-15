from datetime import datetime
from re import search
from typing import Dict, Generator, List
from urllib.parse import urlencode

import pytest
from fastapi import status
from sqlalchemy.orm import Session
from starlette.status import HTTP_200_OK
from starlette.testclient import TestClient

from mariner.core.config import get_app_settings
from mariner.entities.dataset import Dataset
from mariner.entities.user import User
from mariner.schemas.experiment_schemas import Experiment
from mariner.schemas.model_schemas import Model
from tests.fixtures.dataset import setup_create_dataset, teardown_create_dataset
from tests.fixtures.experiments import setup_experiments, teardown_experiments
from tests.fixtures.model import setup_create_model, teardown_create_model
from tests.fixtures.user import get_random_test_user, get_test_user
from tests.utils.user import authentication_token_from_email
from tests.utils.utils import assert_all_is_number, random_lower_string


@pytest.fixture(scope="module")
def user_fixture(db: Session) -> User:
    return get_random_test_user(db)


@pytest.fixture(scope="module")
def user_headers_fixture(
    client: TestClient, db: Session, user_fixture: User
) -> Dict[str, str]:
    return authentication_token_from_email(
        client=client, email=user_fixture.email, db=db
    )


@pytest.fixture(scope="module")
def user_dataset_fixture(
    client: TestClient, db: Session, user_headers_fixture: Dict[str, str]
):
    ds = setup_create_dataset(
        db, client, user_headers_fixture, name=random_lower_string()
    )
    assert ds is not None
    yield ds
    teardown_create_dataset(db, ds)


@pytest.fixture(scope="module")
def user_model_fixture(
    client: TestClient,
    db: Session,
    user_headers_fixture: Dict[str, str],
    user_dataset_fixture: Dataset,
):
    model = setup_create_model(
        client,
        user_headers_fixture,
        dataset_name=user_dataset_fixture.name,
        model_type="regressor",
    )
    yield model
    teardown_create_model(db, model)


@pytest.mark.long
@pytest.mark.integration
def test_post_experiments(
    client: TestClient, mocked_experiment_payload: dict, user_headers_fixture: dict
):
    res = client.post(
        f"{get_app_settings('server').host}/api/v1/experiments/",
        json=mocked_experiment_payload,
        headers=user_headers_fixture,
    )
    assert res.status_code == HTTP_200_OK, res.json()


@pytest.fixture(scope="module")
def user_experiments_fixture(
    db: Session, user_model_fixture: Model
) -> Generator[List[Experiment], None, None]:
    exps = setup_experiments(db, user_model_fixture, num_experiments=3)
    assert len(exps) == 3, "failed in setup of some_experiments fixture"
    yield exps
    teardown_experiments(db, exps)


@pytest.mark.integration  # fixtures use other services
def test_get_experiments(
    client: TestClient,
    user_model_fixture,
    user_experiments_fixture,
    user_headers_fixture,
):
    params = {"modelId": user_model_fixture.id}
    res = client.get(
        f"{get_app_settings('server').host}/api/v1/experiments/",
        params=params,
        headers=user_headers_fixture,
    )
    assert res.status_code == HTTP_200_OK
    body = res.json()
    exps, total = body["data"], body["total"]
    assert total == len(exps) == len(user_experiments_fixture), "gets all experiments"


@pytest.mark.integration  # fixtures use other services
def test_get_experiments_ordered_by_createdAt_desc_url_encoded(
    client: TestClient,
    user_model_fixture,
    user_experiments_fixture,
    user_headers_fixture,
):
    params = {"modelId": user_model_fixture.id, "orderBy": "-createdAt"}
    querystring = urlencode(params)

    res = client.get(
        f"{get_app_settings('server').host}/api/v1/experiments/?{querystring}",
        params=params,
        headers=user_headers_fixture,
    )
    assert res.status_code == HTTP_200_OK
    body = res.json()
    exps, total = body["data"], body["total"]
    assert (
        len(exps) == len(user_experiments_fixture) == total == 3
    ), "gets all experiments"
    for i in range(len(exps[:-1])):
        current, next = exps[i], exps[i + 1]
        assert datetime.fromisoformat(next["createdAt"]) < datetime.fromisoformat(
            current["createdAt"]
        ), "createdAt descending order is not respected"


@pytest.mark.integration  # fixtures use other services
def test_get_experiments_ordered_by_createdAt_desc(
    client: TestClient,
    user_model_fixture,
    user_experiments_fixture,
    user_headers_fixture,
):
    params = {"modelId": user_model_fixture.id, "orderBy": "-createdAt"}

    res = client.get(
        f"{get_app_settings('server').host}/api/v1/experiments/",
        params=params,
        headers=user_headers_fixture,
    )
    assert res.status_code == HTTP_200_OK
    body = res.json()
    exps, total = body["data"], body["total"]
    assert (
        len(exps) == len(user_experiments_fixture) == total == 3
    ), "gets all experiments"
    for i in range(len(exps[:-1])):
        current, next = exps[i], exps[i + 1]
        assert datetime.fromisoformat(next["createdAt"]) < datetime.fromisoformat(
            current["createdAt"]
        ), "createdAt descending order is not respected"


@pytest.mark.integration  # fixtures use other services
def test_get_experiments_ordered_by_createdAt_asc_url_encoded(
    client: TestClient,
    user_model_fixture,
    user_experiments_fixture,
    user_headers_fixture,
):
    params = {"modelId": user_model_fixture.id, "orderBy": "+createdAt"}
    querystring = urlencode(params)

    res = client.get(
        f"{get_app_settings('server').host}/api/v1/experiments/?{querystring}",
        headers=user_headers_fixture,
    )
    assert res.status_code == HTTP_200_OK
    body = res.json()
    exps, total = body["data"], body["total"]
    assert (
        len(exps) == len(user_experiments_fixture) == total == 3
    ), "gets all experiments"
    for i in range(len(exps[:-1])):
        current, next = exps[i], exps[i + 1]
        assert datetime.fromisoformat(next["createdAt"]) > datetime.fromisoformat(
            current["createdAt"]
        ), "createdAt ascending order is not respected"


@pytest.mark.integration  # fixtures use other services
def test_get_experiments_by_stage(
    client: TestClient,
    user_model_fixture,
    user_experiments_fixture,
    user_headers_fixture,
):
    params = {
        "modelId": user_model_fixture.id,
        "page": 0,
        "perPage": 15,
        "stage": ["SUCCESS"],
    }
    res = client.get(
        f"{get_app_settings('server').host}/api/v1/experiments",
        params=params,
        headers=user_headers_fixture,
    )
    body = res.json()
    assert res.status_code == HTTP_200_OK, "Request failed with body: %r" % body
    exps, total = body["data"], body["total"]
    for exp in exps:
        assert exp["stage"] == "SUCCESS", "experiment out of stage filter"

    assert (
        len(exps) == total == 2
    ), "request failed to get the 2 out 3 successfull experiments"


@pytest.mark.integration
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
        f"{get_app_settings('server').host}/api/v1/experiments/epoch_metrics",
        json=metrics_update,
    )
    assert res.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.integration  # fixtures use other services
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
        f"{get_app_settings('server').host}/api/v1/experiments/epoch_metrics",
        json=metrics_update,
        headers={
            "Authorization": f"Bearer {get_app_settings('secrets').application_secret}"
        },
    )
    assert res.status_code == status.HTTP_200_OK


@pytest.mark.integration
def test_get_experiments_metrics_for_model_version(
    client: TestClient,
    user_model_fixture,
    user_experiments_fixture,
    user_headers_fixture,
):
    model_version_id = user_model_fixture.versions[0].id

    res = client.get(
        f"{get_app_settings('server').host}/api/v1/experiments/{model_version_id}/metrics",
        headers=user_headers_fixture,
    )

    assert res.status_code == HTTP_200_OK, "Request failed with body: %r" % res.json()

    experiments_metrics: List[dict] = res.json()
    assert len(experiments_metrics) == len(
        user_experiments_fixture
    ), "Number of experiments with metrics doesn't match the number of user experiments"

    for experiment_metrics in experiments_metrics:
        assert (
            experiment_metrics["modelVersionId"] == model_version_id
        ), "Experiment metrics have incorrect model version id"

        if experiment_metrics["stage"] == "SUCCESS":
            assert isinstance(
                experiment_metrics["history"], dict
            ), "Experiment metrics are not present for successful experiment"

            for key, value in experiment_metrics["history"].items():
                assert key == "epochs" or bool(
                    search(r"^(train|val|test)\/\w+\/", key)
                ), f"Invalid metric name: {key}"

                assert_all_is_number(value)
