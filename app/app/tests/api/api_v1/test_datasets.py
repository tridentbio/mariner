import json
from datetime import datetime
from typing import Dict

from fastapi.testclient import TestClient
from sqlalchemy.orm.session import Session
from starlette import status

from app.core.config import settings
from app.features.dataset.crud import repo
from app.features.dataset.model import Dataset as DatasetModel
from app.features.dataset.schema import DatasetCreateRepo, Split
from app.tests.conftest import get_test_user, mock_dataset
from app.tests.utils.utils import random_lower_string


def test_get_my_datasets(
    client: TestClient, normal_user_token_headers: Dict[str, str]
) -> None:
    r = client.get(f"{settings.API_V1_STR}/datasets", headers=normal_user_token_headers)
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload["total"], int)
    assert isinstance(payload["data"], list)


def test_post_datasets(
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    db: Session,
) -> None:
    metadatas = [
        {
            "pattern": "exp",
            "data_type": {
                "domain_kind": "numerical",
            },
            "description": "speriment measurement",
        },
        {
            "pattern": "smiles",
            "data_type": {
                "domain_kind": "smiles",
            },
            "description": "SMILES representaion of molecule",
        },
    ]

    with open("app/tests/data/HIV.csv", "rb") as f:
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
        assert len(response["columnsMetadata"]) == 2
        ds = repo.get(db, id)
        assert ds is not None
        assert ds.name == response["name"]
        assert len(ds.columns_metadata) == 2


def test_post_datasets_name_conflict(
    client: TestClient,
    some_dataset: DatasetModel,
    normal_user_token_headers: dict[str, str],
):
    ds = mock_dataset(name=some_dataset.name)
    with open("app/tests/data/zinc.csv", "rb") as f:
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
    new_name = "new name"
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
    updated = repo.get(db, response["id"])
    updated = db.query(DatasetModel).filter(DatasetModel.id == some_dataset.id).first()
    assert updated is not None
    assert updated.id == response["id"]
    assert updated.name == new_name


def test_delete_datasets(
    client: TestClient, normal_user_token_headers: Dict[str, str], db: Session
) -> None:
    user = get_test_user(db)
    dataset = repo.create(
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
    ds = repo.get(db, dataset.id)
    assert ds is None


def test_get_csv_metadata(
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
) -> None:
    with open("app/tests/data/Lipophilicity.csv", "rb") as f:
        res = client.post(
            f"{settings.API_V1_STR}/datasets/csv-metadata",
            files={"file": ("dataset.csv", f.read())},
            headers=normal_user_token_headers,
        )
        assert res.status_code == status.HTTP_200_OK
        cols = res.json()
        assert isinstance(cols, list)
        assert len(cols) == 3
        colnames = [item["name"] for item in cols]
        assert "CMPD_CHEMBLID" in colnames
        assert "exp" in colnames
        assert "smiles" in colnames
