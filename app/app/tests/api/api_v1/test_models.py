import pytest
import io
import mlflow.pyfunc

from fastapi.datastructures import UploadFile
from sqlalchemy.orm.session import Session
from starlette.status import HTTP_200_OK
from starlette.testclient import TestClient
from app.core.mlflowapi import get_deployment_plugin

from app.features.user.model import User

from app.core.config import settings
from app.features.user.crud import repo as user_repo
from app.features.user.model import User
from app.features.model import controller as model_ctl
from app.features.model.schema.model import ModelCreate
from app.tests.utils.utils import random_lower_string

def get_test_user(db: Session) -> User:
    user = user_repo.get_by_email(db, email=settings.EMAIL_TEST_USER)
    assert user is not None
    return user


def mock_model(created_by: User) -> ModelCreate:
    return ModelCreate(
        name=random_lower_string(),
        model_description=random_lower_string(),
        model_version_description=random_lower_string(),
        created_by_id=created_by.id
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
    with open(model_path, 'rb') as f:
        res = client.post(
            f"{settings.API_V1_STR}/models/",
            data=data,
            files={"file": ("model.pt", f)},
            headers=headers,
        )
        assert res.status_code == HTTP_200_OK
    return res


def test_post_models_success(
    db: Session,
    client: TestClient,
    normal_user_token_headers: dict[str, str],
):
    model_path = "app/tests/data/model.pt"
    user = get_test_user(db)
    model = mock_model(user)
    model_name = model.name
    model_description = model.model_description
    model_version_description = model.model_version_description
    assert user is not None
    with open(model_path, "rb") as f:
        res = client.post(
            f"{settings.API_V1_STR}/models/",
            data={
                "name": model_name,
                "description": model_description,
                "version_description": model_version_description
            },
            files={"file": ("model.pt", f)},
            headers=normal_user_token_headers,
        )
        body = res.json()
        assert res.status_code == HTTP_200_OK
        assert body["name"] == model_name
        assert body["createdById"] == user.id
        assert body["modelDescription"] == model_description
        assert body["modelVersionDescription"] == model_version_description
        assert len(body["latestVersions"]) == 1
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/1")
        assert model is not None

def setup_function():
    pass


def test_get_models_success(
    client: TestClient, normal_user_token_headers: dict[str, str],
    db: Session
):
    setup_create_model(db, client, headers=normal_user_token_headers)
    res = client.get(
        f"{settings.API_V1_STR}/models/", headers=normal_user_token_headers
    )
    assert res.status_code == HTTP_200_OK
    user = get_test_user(db)
    body = res.json()
    models = body["data"]
    total = body['total']
    assert len(models) > 0
    assert total > 0
    for model in models:
        print(model)
        assert model['createdById'] == user.id


def test_post_models_deployment(db: Session, client: TestClient, normal_user_token_headers: dict[str, str]):
    res = setup_create_model(db, client, headers=normal_user_token_headers)
    model = res.json()
    data = {
        'name': random_lower_string(),
        'model_name': model['name'],
        'model_version': int(model['latestVersions'][-1]['version']),
    }
    res = client.post(
        f"{settings.API_V1_STR}/deployments/",
        json=data,
        headers=normal_user_token_headers
    )
    assert res.status_code == HTTP_200_OK
    plugin = get_deployment_plugin()
    assert len(plugin.list_endpoints()) >= 1


def test_add_version_to_model():
   pass


def test_update_model():
    pass


def test_delete_model():
    pass


