from typing import List

import mlflow.pyfunc
import pandas as pd
from pydantic.networks import AnyHttpUrl
from sqlalchemy.orm.session import Session
from starlette.status import HTTP_200_OK
from starlette.testclient import TestClient

from app.core.config import settings
from app.core.mlflowapi import get_deployment_plugin
from app.features.model import generate
from app.features.model.model import ModelVersion
from app.features.model.schema.model import Model
from app.tests.conftest import get_test_user, mock_model
from app.tests.utils.utils import random_lower_string


def test_post_models_success(
    db: Session,
    client: TestClient,
    normal_user_token_headers: dict[str, str],
):
    user = get_test_user(db)
    model = mock_model()
    res = client.post(
        f"{settings.API_V1_STR}/models/",
        json=model.dict(),
        headers=normal_user_token_headers,
    )
    body = res.json()

    assert res.status_code == HTTP_200_OK
    assert body["name"] == model.name
    assert body["createdById"] == user.id
    assert body["description"] == model.model_description
    assert "versions" in body
    assert len(body["versions"]) == 1
    version = body["versions"][0]
    model_version = version["modelVersion"]
    assert version["config"]["name"] is not None
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model.name}/{model_version}")
    assert model is not None
    db_model_config = db.query(ModelVersion).filter(
        ModelVersion.model_name == body["name"]
        and ModelVersion.model_version == model_version
    )
    assert db_model_config is not None


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
        assert model["createdById"] == user.id


def test_post_models_deployment(
    client: TestClient, normal_user_token_headers: dict[str, str], some_model: Model
):
    data = {
        "name": random_lower_string(),
        "modelName": some_model.name,
        "modelVersion": int(some_model.versions[-1].model_version),
    }
    res = client.post(
        f"{settings.API_V1_STR}/deployments/",
        json=data,
        headers=normal_user_token_headers,
    )
    assert res.status_code == HTTP_200_OK
    body = res.json()
    assert body["modelName"] == data["modelName"]
    plugin = get_deployment_plugin()
    assert len(plugin.list_deployments()) >= 1


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

    def assert_component_info(component_dict: dict):
        assert "docs" in component_dict
        assert "docsLink" in component_dict
        assert isinstance(component_dict["docs"], str)
        assert AnyHttpUrl(component_dict["docs"], scheme="https") is not None

    for comp in generate.layers:
        assert comp.name in layer_types
    for comp in generate.featurizers:
        assert comp.name in featurizer_types

    for layer_payload in payload["componentAnnotations"]:
        assert_component_info(layer_payload)


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
    model_version = some_model.versions[-1].model_version
    route = (
        f"{settings.API_V1_STR}/models/{user_id}/{model_name}/{model_version}/predict"
    )
    df = pd.DataFrame(
        {
            "smiles": [
                "CCCC",
                "CCCCC",
                "CCCCCCC",
            ],
            "mwt": [0.3, 0.1, 0.9],
            "tpsa": [0.3, 0.1, 0.9],
        }
    )
    data = df.to_json()
    res = client.post(route, data, headers=normal_user_token_headers)
    assert res.status_code == 200
