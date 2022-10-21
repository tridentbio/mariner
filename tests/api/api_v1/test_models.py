from typing import Any, List

import mlflow.pyfunc
import pytest
from mockito import patch
from pydantic.networks import AnyHttpUrl
from sqlalchemy.orm.session import Session
from starlette import status
from starlette.status import HTTP_200_OK
from starlette.testclient import TestClient

from mariner.core.config import settings
from mariner.core.mlflowapi import get_deployment_plugin
from mariner.entities import (
    Dataset as DatasetEntity,
    Model as ModelEntity,
    ModelVersion,
)
from mariner.schemas.dataset_schemas import QuantityDataType
from model_builder import generate
from model_builder import layers_schema as layers
from model_builder.schemas import (
    ColumnConfig,
    DatasetConfig,
    ModelSchema,
)
from mariner.schemas.model_schemas import Model, ModelCreate
from tests.conftest import mock_model, setup_create_model, get_test_user
from tests.utils.utils import random_lower_string


@pytest.fixture(scope="function")
def mocked_invalid_model(some_dataset: DatasetEntity) -> ModelCreate:
    config = ModelSchema(
        name=random_lower_string(),
        dataset=DatasetConfig(
            name=some_dataset.name,
            feature_columns=[
                ColumnConfig(
                    name="mwt",
                    data_type=QuantityDataType(domain_kind="numeric", unit="mole"),
                )
            ],
            target_column=ColumnConfig(
                name="tpsa",
                data_type=QuantityDataType(domain_kind="numeric", unit="mole"),
            ),
        ),
        featurizers=[],
        layers=[
            layers.TorchlinearLayerConfig(
                type="torch.nn.Linear",
                args=layers.TorchlinearArgs(in_features=27, out_features=1),
                input="mwt",
                name="1",
            )
        ],
    )
    model = ModelCreate(
        name=config.name,
        model_description=random_lower_string(),
        model_version_description=random_lower_string(),
        config=config,
    )
    return model


def test_post_models_success(
    db: Session,
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    some_dataset: DatasetEntity,
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
    assert "columns" in body
    assert body["createdById"] == user.id
    assert body["description"] == model.model_description
    assert "versions" in body
    assert len(body["versions"]) == 1
    version = body["versions"][0]
    mlflow_model_name = version["mlflowModelName"]
    mlflow_version = version["mlflowVersion"]
    model_version = version["name"]
    assert version["config"]["name"] is not None
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{mlflow_model_name}/{mlflow_version}"
    )
    assert model is not None
    db_model_config = db.query(ModelVersion).filter(
        ModelVersion.id == version["id"] and ModelVersion.name == model_version
    )
    assert db_model_config is not None


def test_post_models_on_existing_model_creates_new_version(
    db: Session,
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    some_model: Model,
):
    new_version = some_model.versions[0].copy()
    res = client.post(
        f"{settings.API_V1_STR}/models/",
        json={
            "name": some_model.name,
            "config": new_version.config.dict(),
            "modelVersionDescription": "This should be  version 2",
        },
        headers=normal_user_token_headers,
    )
    assert res.status_code == status.HTTP_200_OK
    model = db.query(ModelEntity).filter(ModelEntity.name == some_model.name).first()
    assert model
    model = Model.from_orm(model)
    assert len(model.versions) == 2
    assert model.versions[-1].name


def test_post_models_dataset_not_found(
    client: TestClient, normal_user_token_headers: dict[str, str], some_model: Model
):
    model = mock_model(some_model.name)
    model.name = random_lower_string()
    datasetname = "a dataset name that is not registered"
    model.config.dataset.name = datasetname
    res = client.post(
        f"{settings.API_V1_STR}/models/",
        json=model.dict(),
        headers=normal_user_token_headers,
    )
    assert res.status_code == status.HTTP_404_NOT_FOUND
    assert res.json()["detail"] == f'Dataset "{datasetname}" not found'


def test_post_models_check_model_name_is_unique(
    client: TestClient, superuser_token_headers: dict[str, str], some_model: Model
):
    model = mock_model(some_model.name)
    res = client.post(
        f"{settings.API_V1_STR}/models/",
        json=model.dict(),
        headers=superuser_token_headers,
    )
    assert res.status_code == status.HTTP_409_CONFLICT
    assert res.json()["detail"] == "Another model is already registered with that name"


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


@pytest.mark.skip("Flaky test not so important now")
def test_post_models_deployment(
    client: TestClient, normal_user_token_headers: dict[str, str], some_model: Model
):
    data = {
        "name": random_lower_string(),
        "modelName": some_model.name,
        "modelVersion": int(some_model.versions[-1].name),
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


def test_delete_model(
    client: TestClient,
    db: Session,
    normal_user_token_headers: dict[str, str],
    some_dataset: DatasetEntity,
):
    model = setup_create_model(client, normal_user_token_headers, some_dataset)
    res = client.delete(
        f"{settings.API_V1_STR}/models/{model.id}", headers=normal_user_token_headers
    )
    assert res.status_code == 200
    assert not db.query(ModelEntity).filter(ModelEntity.id == model.id).first()


def test_post_predict(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    some_model: Model,
):
    model_version = some_model.versions[-1].id
    route = f"{settings.API_V1_STR}/models/{model_version}/predict"
    res = client.post(
        route,
        json={
            "smiles": [
                "CCCC",
                "CCCCC",
                "CCCCCCC",
            ],
            "mwt": [0.3, 0.1, 0.9],
        },
        headers=normal_user_token_headers,
    )
    assert res.status_code == 200
    body = res.json()
    assert len(body) == 3


def test_get_model_version(
    client: TestClient, some_model: Model, normal_user_token_headers: dict[str, str]
):
    res = client.get(
        f"{settings.API_V1_STR}/models/{some_model.id}/",
        headers=normal_user_token_headers,
    )
    assert res.status_code == 200
    body = res.json()
    assert body["name"] == some_model.name


def test_post_models_invalid_type(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    mocked_invalid_model: ModelCreate,
):
    wrong_layer_name = mocked_invalid_model.config.layers[0].name
    mocked_invalid_model.config.layers[0].type = "aksfkasmf"
    res = client.post(
        f"{settings.API_V1_STR}/models/",
        headers=normal_user_token_headers,
        json=mocked_invalid_model.dict(),
    )
    error_body = res.json()
    assert res.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "detail" in error_body
    assert len(error_body["detail"]) == 1
    assert error_body["detail"][0]["type"] == "value_error.unknowncomponenttype"
    assert error_body["detail"][0]["ctx"]["component_name"] == wrong_layer_name


def test_post_models_missing_arguments(
    client: TestClient,
    mocked_invalid_model: ModelCreate,
    normal_user_token_headers: dict[str, str],
):

    tmp = mocked_invalid_model.dict()
    del tmp["config"]["layers"][0]["args"]["in_features"]
    mocked_invalid_model = ModelCreate.construct(**tmp)
    wrong_layer_name = mocked_invalid_model.config["layers"][0]["name"]
    res = client.post(
        f"{settings.API_V1_STR}/models/",
        headers=normal_user_token_headers,
        json=mocked_invalid_model.dict(),
    )
    assert res.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    error_body = res.json()
    assert "detail" in error_body
    assert len(error_body["detail"]) == 1
    assert error_body["detail"][0]["type"] == "value_error.missingcomponentargs"
    assert error_body["detail"][0]["ctx"]["component_name"] == wrong_layer_name


def test_post_check_config_good_model(
    some_dataset: DatasetEntity,
    normal_user_token_headers: dict[str, str],
    client: TestClient,
    model_config: ModelSchema,
):
    res = client.post(
        f"{settings.API_V1_STR}/models/check-config",
        headers=normal_user_token_headers,
        json=model_config.dict(),
    )
    assert res.status_code == 200
    body = res.json()
    assert "output" in body


def test_post_check_config_bad_model(
    some_dataset: DatasetEntity,
    normal_user_token_headers: dict[str, str],
    client: TestClient,
    model_config: ModelSchema,
):
    import model_builder.model

    def raise_(x: Any):
        raise Exception("some random exception in the forward")

    with patch(model_builder.model.CustomModel.forward, raise_):
        res = client.post(
            f"{settings.API_V1_STR}/models/check-config",
            headers=normal_user_token_headers,
            json=model_config.dict(),
        )
        assert res.status_code == 200
        body = res.json()
        assert not body["output"]
        assert "stackTrace" in body
        assert "some random exception in the forward" in body["stackTrace"]
        assert len(body["stackTrace"].split("\n")) > 1


def test_get_name_suggestion(
    client: TestClient, normal_user_token_headers: dict[str, str]
):
    res = client.get(
        f"{settings.API_V1_STR}/models/name-suggestion",
        headers=normal_user_token_headers,
    )
    assert res.status_code == 200
    body = res.json()
    assert "name" in body
    assert type(body["name"]) == str
    assert len(body["name"]) > 4


def test_model_versioning():
    """
    Checks if the model versioning mapping between mariner
    models and MLFlow Registry is correct
    """
    pass
