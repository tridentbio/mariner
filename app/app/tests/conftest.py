import json
from typing import Dict, Generator, Optional

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from starlette import status

from app.core.aws import Bucket, delete_s3_file
from app.core.config import settings
from app.db.session import SessionLocal
from app.features.dataset.model import Dataset
from app.features.model.model import Model as ModelEntity
from app.features.user.crud import repo as user_repo
from app.features.user.model import User
from app.main import app
from app.tests.features.model.conftest import setup_create_model
from app.tests.utils.user import authentication_token_from_email
from app.tests.utils.utils import get_superuser_token_headers


@pytest.fixture(scope="session")
def db() -> Generator:
    sesion = SessionLocal()
    yield sesion


@pytest.fixture(scope="module")
def client() -> Generator:
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def superuser_token_headers(client: TestClient) -> Dict[str, str]:
    return get_superuser_token_headers(client)


@pytest.fixture(scope="module")
def normal_user_token_headers(client: TestClient, db: Session) -> Dict[str, str]:
    return authentication_token_from_email(
        client=client, email=settings.EMAIL_TEST_USER, db=db
    )


def get_test_user(db: Session) -> User:
    user = user_repo.get_by_email(db, email=settings.EMAIL_TEST_USER)
    assert user is not None
    return user


# DATASET GLOBAL FIXTURES
def mock_dataset(name: Optional[str] = None):
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
        "name": name if name else "Small Zinc dataset",
        "description": "Test description",
        "splitType": "random",
        "splitTarget": "60-20-20",
        "columnsDescriptions": json.dumps(descriptions),
        "columnsMetadata": json.dumps(metadatas),
    }


def setup_create_dataset(
    db: Session, client: TestClient, normal_user_token_headers: Dict[str, str]
):
    data = mock_dataset()
    db.query(Dataset).filter(Dataset.name == data["name"]).delete()
    db.commit()
    with open("app/tests/data/zinc.csv", "rb") as f:
        res = client.post(
            f"{settings.API_V1_STR}/datasets/",
            data=data,
            files={"file": ("zinc.csv", f.read())},
            headers=normal_user_token_headers,
        )
        assert res.status_code == status.HTTP_200_OK
        body = res.json()
        return db.query(Dataset).get(body["id"])


def teardown_create_dataset(db: Session, dataset: Dataset):
    ds = db.query(Dataset).get(dataset.id)
    assert ds is not None
    db.delete(ds)
    db.commit()
    try:
        delete_s3_file(bucket=Bucket.Datasets, key=ds.data_url)
    except Exception:
        pass


@pytest.fixture(scope="module")
def some_dataset(
    db: Session, client: TestClient, normal_user_token_headers: Dict[str, str]
):
    ds = setup_create_dataset(db, client, normal_user_token_headers)
    assert ds is not None
    yield ds
    teardown_create_dataset(db, ds)


# MODEL GLOBAL FIXTURES


@pytest.fixture(scope="module")
def some_model(
    db: Session,
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    some_dataset: Dataset,
):
    db.query(ModelEntity).delete()
    db.commit()
    model = setup_create_model(client, normal_user_token_headers, some_dataset)
    yield model
    # teardown_create_model(db, model.name)


def mock_dataset_item():
    import torch
    from torch_geometric.data import Data

    x = torch.ones(21, 26, dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    mwt = torch.tensor([[230.0]], dtype=torch.float)

    dataset_input = {
        "MolToGraphFeaturizer": Data(x=x, edge_index=edge_index, batch=None),
        "mwt": mwt,
        "tpsa": mwt,
    }

    return dataset_input


@pytest.fixture(scope="module")
def dataset_sample():
    return mock_dataset_item()
