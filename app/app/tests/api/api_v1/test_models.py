from typing import Dict, List

import mlflow.pyfunc
import pandas as pd
import pytest
from mlflow.tracking.client import MlflowClient
from sqlalchemy.orm.session import Session
from starlette.status import HTTP_200_OK
from starlette.testclient import TestClient

from app.core.config import settings
from app.core.mlflowapi import get_deployment_plugin
from app.features.model import generate
from app.features.model.model import Model as ModelEntity
from app.features.model.schema.model import Model, ModelCreate
from app.features.user.crud import repo as user_repo
from app.features.user.model import User
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
    mlflowclient = MlflowClient()
    mlflowclient.delete_registered_model(model_name)


@pytest.fixture()
def some_model(
    db: Session, client: TestClient, normal_user_token_headers: Dict[str, str]
):
    model = setup_create_model(db, client, normal_user_token_headers)
    yield model
    teardown_create_model(db, model.name)


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
                "version_description": model_version_description,
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


def test_get_models_success(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    db: Session,
    some_model: Model,
):
    res = client.get(
        f"{settings.API_V1_STR}/models/", headers=normal_user_token_headers
    )
    assert res.status_code == HTTP_200_OK
    user = get_test_user(db)
    body = res.json()
    models = body["data"]
    total = body["total"]
    assert len(models) > 0
    assert total > 0
    for model in models:
        print(model)
        assert model["createdById"] == user.id


def test_post_models_deployment(
    client: TestClient, normal_user_token_headers: dict[str, str], some_model: Model
):
    data = {
        "name": random_lower_string(),
        "model_name": some_model.name,
        "model_version": int(some_model.latest_versions[-1].version),
    }
    res = client.post(
        f"{settings.API_V1_STR}/deployments/",
        json=data,
        headers=normal_user_token_headers,
    )
    assert res.status_code == HTTP_200_OK
    plugin = get_deployment_plugin()
    assert len(plugin.list_endpoints()) >= 1


def test_get_model_options(
    client: TestClient, normal_user_token_headers: dict[str, str]
):
    res = client.get(
        f"{settings.API_V1_STR}/models/options", headers=normal_user_token_headers
    )
    assert res.status_code == HTTP_200_OK
    payload = res.json()
    assert "layers" in payload
    assert "featurizers" in payload
    assert len(payload["layers"]) > 0
    assert len(payload["featurizers"]) > 0
    layer_types: List[str] = [layer["type"] for layer in payload["layers"]]
    featurizer_types: List[str] = [layer["type"] for layer in payload["featurizers"]]
    for comp in generate.layers:
        assert comp.name in layer_types
    for comp in generate.featurizers:
        assert comp.name in featurizer_types


def test_add_version_to_model():
    pass


def test_update_model():
    pass


def test_delete_model():
    pass


def test_post_predict(
    db: Session,
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    some_model: Model,
):
    user_id = get_test_user(db).id
    model_name = some_model.name
    model_version = some_model.latest_versions[-1]["version"]
    route = (
        f"{settings.API_V1_STR}/models/{user_id}/{model_name}/{model_version}/predict"
    )
    df = pd.DataFrame(
        {
            "smiles": [
                1,
                2,
                3,
            ]
        }
    )
    # data = {
    #     'model_input': df.to_json()
    # }
    data = df.to_json()
    res = client.post(route, data, headers=normal_user_token_headers)
    assert res.status_code == 200
