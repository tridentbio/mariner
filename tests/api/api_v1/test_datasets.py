import json
from datetime import datetime
from typing import Dict

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm.session import Session
from starlette import status

from mariner.core.config import settings
from mariner.entities import Dataset as DatasetModel
from mariner.schemas.dataset_schemas import DatasetCreateRepo, Split
from mariner.stores.dataset_sql import dataset_store
from tests.fixtures.dataset import mock_dataset
from tests.fixtures.user import get_test_user
from tests.utils.utils import random_lower_string


def test_get_my_datasets(
    client: TestClient, normal_user_token_headers: Dict[str, str]
) -> None:
    r = client.get(f"{settings.API_V1_STR}/datasets", headers=normal_user_token_headers)
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload["total"], int)
    assert isinstance(payload["data"], list)


@pytest.mark.long
def test_post_datasets(
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    db: Session,
) -> None:
    metadatas = [
        {
            "pattern": "exp",
            "data_type": {"domain_kind": "numeric", "unit": "mole"},
            "description": "experiment measurement",
            "unit": "mole",
        },
        {
            "pattern": "smiles",
            "data_type": {
                "domain_kind": "smiles",
            },
            "description": "SMILES representaion of molecule",
        },
    ]

    with open("tests/data/Lipophilicity.csv", "rb") as f:
        res = client.post(
            f"{settings.API_V1_STR}/datasets/",
            data={
                "name": random_lower_string(),
                "description": "Test description",
                "splitType": "random",
                "splitTarget": "60-20-20",
                "columnsMetadata": json.dumps(metadatas),
            },
            files={"file": ("dataset.csv", f.read())},
            headers=normal_user_token_headers,
        )
        assert res.status_code == status.HTTP_200_OK
        response = res.json()
        id = response["id"]
        assert response["readyStatus"] == "processing"

        with client.websocket_connect(
            "/ws?token=" + normal_user_token_headers["Authorization"].split(" ")[1],
            timeout=60,
        ) as ws:
            message = ws.receive_json()
            assert message is not None
            assert message["type"] == "dataset-process-finish"
            assert "created successfully" in message["data"].get("message", "")

        ds = dataset_store.get(db, id)
        assert ds is not None
        assert ds.name == response["name"]
        assert response["columns"] == 3
        assert len(ds.columns_metadata) == 2


def test_post_datasets_name_conflict(
    client: TestClient,
    some_dataset: DatasetModel,
    normal_user_token_headers: dict[str, str],
):
    ds = mock_dataset(name=some_dataset.name)
    with open("tests/data/zinc.csv", "rb") as f:
        res = client.post(
            f"{settings.API_V1_STR}/datasets/",
            data=ds,
            files={"file": ("zinc.csv", f.read())},
            headers=normal_user_token_headers,
        )
        assert res.status_code == status.HTTP_409_CONFLICT


def test_put_datasets(
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    db: Session,
    some_dataset: DatasetModel,
) -> None:
    new_name = random_lower_string()
    r = client.put(
        f"{settings.API_V1_STR}/datasets/{some_dataset.id}",
        data={
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
    db.commit()
    updated = dataset_store.get(db, response["id"])
    updated = db.query(DatasetModel).filter(DatasetModel.id == some_dataset.id).first()
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
        f"{settings.API_V1_STR}/datasets/{dataset.id}",
        headers=normal_user_token_headers,
    )
    assert r.status_code == status.HTTP_200_OK
    ds = dataset_store.get(db, dataset.id)
    assert ds is None


def test_get_csv_metadata(
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
) -> None:
    with open("tests/data/Lipophilicity.csv", "rb") as f:
        res = client.post(
            f"{settings.API_V1_STR}/datasets/csv-metadata",
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
