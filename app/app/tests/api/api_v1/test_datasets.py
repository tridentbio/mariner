from datetime import datetime
from typing import Dict

from fastapi.testclient import TestClient
from sqlalchemy.orm.session import Session
from starlette import status

from app.core.config import settings
from app.features.dataset.crud import repo
from app.features.dataset.schema import DatasetCreateRepo, Split


def test_get_my_datasets(
    client: TestClient, normal_user_token_headers: Dict[str, str]
) -> None:
    r = client.get(f"{settings.API_V1_STR}/datasets", headers=normal_user_token_headers)
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload["total"], int)
    assert isinstance(payload["data"], list)


def test_create_dataset(
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    db: Session,
) -> None:
    with open("app/tests/data/HIV.csv", "rb") as f:
        res = client.post(
            f"{settings.API_V1_STR}/datasets/",
            data={
                "name": "Test dataset",
                "description": "Test description",
                "splitType": "random",
                "splitTarget": "60-20-20",
            },
            files={"file": ("dataset.csv", f.read())},
            headers=normal_user_token_headers,
        )
        assert res.status_code == status.HTTP_200_OK
        id = res.json()["id"]
        ds = repo.get(db, id)
        assert ds != None


def test_update_dataset(
    client: TestClient, superuser_token_headers: Dict[str, str], db: Session
) -> None:
    dataset = repo.create(
        db,
        DatasetCreateRepo(
            bytes=30,
            name="test",
            stats="",
            data_url="",
            created_at=datetime.now(),
            created_by_id=1,
            description="test",
            rows=100,
            columns=5,
            split_type="random",
            split_actual=None,
            split_target=Split("60-20-20"),
        ),
    )
    new_name = "new name"
    r = client.put(
        f"{settings.API_V1_STR}/datasets/{dataset.id}",
        data={
            "name": new_name,
            "description": new_name,
            "splitType": "random",
            "splitTarget": "60-20-20",
        },
        headers=superuser_token_headers,
    )
    assert r.status_code == status.HTTP_200_OK
    updated = repo.get(db, dataset.id)
    assert updated != None
    assert updated.name == new_name


def test_delete_dataset(
    client: TestClient, superuser_token_headers: Dict[str, str], db: Session
) -> None:
    dataset = repo.create(
        db,
        DatasetCreateRepo(
            bytes=30,
            name="test",
            stats="",
            data_url="",
            created_at=datetime.now(),
            created_by_id=1,
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
        headers=superuser_token_headers,
    )
    assert r.status_code == status.HTTP_200_OK
    ds = repo.get(db, dataset.id)
    assert ds == None
