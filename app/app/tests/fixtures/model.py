from typing import Dict

import mlflow.tracking
import pytest
from sqlalchemy.orm.session import Session
from starlette.status import HTTP_200_OK
from starlette.testclient import TestClient

from app.core.config import settings
from app.features.model.model import Model as ModelEntity
from app.features.model.schema.model import Model, ModelCreate
from app.features.user.model import User
from app.tests.conftest import get_test_user
from app.tests.utils.utils import random_lower_string


def mock_model(created_by: User) -> ModelCreate:
    return ModelCreate(
        name=random_lower_string(),
        model_description=random_lower_string(),
        model_version_description=random_lower_string(),
        created_by_id=created_by.id,
    )


def setup_create_model(db: Session, client: TestClient, headers):
    user = get_test_user(db)
    model = mock_model(user)
    model_path = "app/tests/data/model.pt"
    data = {
        "name": model.name,
        "description": model.model_description,
        "versionDescription": model.model_version_description,
    }
    with open(model_path, "rb") as f:
        res = client.post(
            f"{settings.API_V1_STR}/models/",
            data=data,
            files={"file": ("model.pt", f)},
            headers=headers,
        )
        assert res.status_code == HTTP_200_OK
    return Model.parse_obj(res.json())


def teardown_create_model(db: Session, model_name: str):
    obj = db.query(ModelEntity).filter(ModelEntity.name == model_name).first()
    db.delete(obj)
    db.commit()
    mlflowclient = mlflow.tracking.MlflowClient()
    mlflowclient.delete_registered_model(model_name)


@pytest.fixture(scope="module")
def some_model(
    db: Session, client: TestClient, normal_user_token_headers: Dict[str, str]
):
    model = setup_create_model(db, client, normal_user_token_headers)
    yield model
    teardown_create_model(db, model.name)

@pytest.fixture(scope="module")
def dataset_sample():
    from torch_geometric.data import Data
    import torch

    x = torch.ones(3, 30, dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    mwt = torch.tensor([[230.], [210.], [410.], [430.], [235.]], dtype=torch.float)
    dataset_input = {
        'MolToGraphFeaturizer': Data(x=x, edge_index=edge_index),
        'mwt': mwt
    }

    dataset_input['MolToGraphFeaturizer'].batch
