from typing import Dict
import pytest
import json
from sqlalchemy.orm.session import Session
from starlette import status
from starlette.testclient import TestClient
from app.core.config import settings
from app.features.dataset.model import Dataset

def mock_dataset():
    descriptions = [
        {
            "pattern": "col*",
            "description": "asdasdas",
        },
        {
            "pattern": "col2*",
            "description": "asdasdas",
        },
    ]
    metadatas = [
        {
            "key": "exp",
            "data_type": "numerical",
        },
        {"key": "smiles", "data_type": "smiles"},
    ]

    return {
        "name": "Test dataset",
        "description": "Test description",
        "splitType": "random",
        "splitTarget": "60-20-20",
        "columnsDescriptions": json.dumps(descriptions),
        "columnsMetadata": json.dumps(metadatas),
    }
def setup_create_dataset(db: Session, client: TestClient, normal_user_token_headers: Dict[str, str]):
    data = mock_dataset()
    with open("app/tests/data/HIV.csv", "rb") as f:
        res = client.post(
            f"{settings.API_V1_STR}/datasets/",
            data=data,
            files={"file": ("dataset.csv", f.read())},
            headers=normal_user_token_headers,
        )
        assert res.status_code == status.HTTP_200_OK
        body = res.json()
        return db.query(Dataset).get(body['id'])
def teardown_create_dataset():
    pass

@pytest.fixture(scope="module")
def some_dataset(db: Session, client: TestClient, normal_user_token_headers: Dict[str, str]):
    ds = setup_create_dataset(db, client, normal_user_token_headers)
    yield ds
    teardown_create_dataset()

