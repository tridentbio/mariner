from datetime import datetime
from typing import Dict

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm.session import Session
from starlette import status

from mariner.core.config import get_app_settings
from mariner.entities import Dataset as DatasetModel
from mariner.schemas.dataset_schemas import DatasetCreateRepo, Split
from mariner.stores.dataset_sql import dataset_store
from mariner.utils import hash_md5
from tests.fixtures.dataset import mock_dataset
from tests.fixtures.user import get_test_user
from tests.utils.dataset import get_post_dataset_data
from tests.utils.utils import random_lower_string


def test_get_my_datasets(
    client: TestClient, normal_user_token_headers: Dict[str, str]
) -> None:
    r = client.get(
        f"{get_app_settings().API_V1_STR}/datasets",
        headers=normal_user_token_headers,
    )
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload["total"], int)
    assert isinstance(payload["data"], list)


def is_close(val1: float, val2: float):
    return abs(val1 - val2) < 0.02


@pytest.mark.integration
def test_post_datasets(
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    db: Session,
) -> None:
    with open("tests/data/csv/Lipophilicity.csv", "rb") as f:
        res = client.post(
            f"{get_app_settings().API_V1_STR}/datasets/",
            data=get_post_dataset_data(),
            files={"file": ("dataset.csv", f.read())},
            headers=normal_user_token_headers,
        )
        assert res.status_code == status.HTTP_200_OK
        response = res.json()
        id = response["id"]
        assert response["readyStatus"] == "processing"

        with client.websocket_connect(
            "/ws?token="
            + normal_user_token_headers["Authorization"].split(" ")[1],
            timeout=60,
        ) as ws:
            message = ws.receive_json()
            assert message is not None
            assert message["type"] == "dataset-process-finish"
            assert "created successfully" in message["data"].get("message", "")

        ds = dataset_store.get(db, id)
        assert is_close(ds.bytes, 1360)
        assert ds is not None
        assert ds.name == response["name"]
        assert ds.columns == 4
        assert len(ds.columns_metadata) == 2


@pytest.mark.long
@pytest.mark.integration
def test_post_datasets_invalid(
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    db: Session,
) -> None:
    with open("tests/data/csv/bad_dataset.csv", "rb") as f:
        res = client.post(
            f"{get_app_settings().API_V1_STR}/datasets/",
            data=get_post_dataset_data(),
            files={"file": ("dataset.csv", f.read())},
            headers=normal_user_token_headers,
        )
        assert res.status_code == status.HTTP_200_OK
        response = res.json()
        id = response["id"]
        assert response["readyStatus"] == "processing"

        with client.websocket_connect(
            "/ws?token="
            + normal_user_token_headers["Authorization"].split(" ")[1],
            timeout=60,
        ) as ws:
            message = ws.receive_json()
            assert message is not None
            assert message["type"] == "dataset-process-finish"
            assert "error on dataset creation" in message["data"].get(
                "message", ""
            )

        ds = dataset_store.get(db, id)
        assert ds.name == response["name"]
        assert ds.errors is not None
        assert len(ds.errors["columns"]) == 2
        assert len(ds.errors["rows"]) == 2
        assert isinstance(ds.errors["dataset_error_key"], str)


@pytest.mark.integration  # requires ray some_dataset
def test_post_datasets_name_conflict(
    client: TestClient,
    some_dataset: DatasetModel,
    normal_user_token_headers: dict[str, str],
):
    ds = mock_dataset(name=some_dataset.name)
    with open("tests/data/csv/zinc.csv", "rb") as f:
        res = client.post(
            f"{get_app_settings().API_V1_STR}/datasets/",
            data=ds,
            files={"file": ("zinc.csv", f.read())},
            headers=normal_user_token_headers,
        )
        assert res.status_code == status.HTTP_409_CONFLICT


@pytest.mark.integration  # requires ray some_dataset
@pytest.mark.skip(reason="taking very long")
def test_put_datasets(
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    db: Session,
    some_dataset: DatasetModel,
) -> None:
    new_name = random_lower_string()
    r = client.put(
        f"{get_app_settings().API_V1_STR}/datasets/{some_dataset.id}",
        json={
            "name": new_name,
            "description": new_name,
            "splitType": "random",
            "splitTarget": "60-20-20",
        },
        headers=normal_user_token_headers,
    )
    assert r.status_code == status.HTTP_200_OK
    response = r.json()
    assert response is not None
    assert response["name"] == new_name

    with client.websocket_connect(
        "/ws?token="
        + normal_user_token_headers["Authorization"].split(" ")[1],
        timeout=60,
    ) as ws:
        message = ws.receive_json()
        assert message is not None
        assert message["type"] == "dataset-process-finish"
        assert "created successfully" in message["data"].get("message", "")

    db.commit()
    updated = dataset_store.get(db, response["id"])
    updated = (
        db.query(DatasetModel)
        .filter(DatasetModel.id == some_dataset.id)
        .first()
    )
    assert updated is not None
    assert updated.id == response["id"]
    assert updated.name == new_name


def test_delete_datasets(
    client: TestClient, normal_user_token_headers: Dict[str, str], db: Session
) -> None:
    user = get_test_user(db)
    dataset = dataset_store.create(
        db,
        DatasetCreateRepo(
            bytes=30,
            name="test",
            stats="{}",
            data_url="",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by_id=user.id,
            description="test",
            rows=100,
            columns=5,
            split_type="random",
            split_actual=None,
            split_target=Split("60-20-20"),
        ),
    )
    r = client.delete(
        f"{get_app_settings().API_V1_STR}/datasets/{dataset.id}",
        headers=normal_user_token_headers,
    )
    assert r.status_code == status.HTTP_200_OK
    ds = dataset_store.get(db, dataset.id)
    assert ds is None


@pytest.mark.integration
def test_get_csv_metadata(
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
) -> None:
    with open("tests/data/csv/Lipophilicity.csv", "rb") as f:
        res = client.post(
            f"{get_app_settings().API_V1_STR}/datasets/csv-metadata",
            files={"file": ("dataset.csv", f.read())},
            headers=normal_user_token_headers,
        )
        assert res.status_code == status.HTTP_200_OK
        cols = res.json()
        assert {
            "name": "exp",
            "dtype": {
                "domainKind": "numeric",
            },
        } in cols
        assert {
            "name": "smiles",
            "dtype": {
                "domainKind": "smiles",
            },
        } in cols


@pytest.mark.parametrize("route", ["file", "file-with-errors"])
@pytest.mark.integration
def test_download_dataset(
    normal_user_token_headers: dict,
    some_dataset_without_process: DatasetModel,
    client: TestClient,
    route: str,
):
    res = client.get(
        f"{get_app_settings().API_V1_STR}/datasets/{some_dataset_without_process.id}/{route}",
        headers=normal_user_token_headers,
    )
    assert res.status_code == 200, f"route datasets/{route} failed"

    hash = hash_md5(data=res.content)
    assert (
        f"datasets/{hash}.csv" == some_dataset_without_process.data_url
    ), f"downloaded file hash does not match in route datasets/{route}"
