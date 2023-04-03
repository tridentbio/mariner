"""
Tests the mariner.models package
"""

import pytest
from pydantic import AnyHttpUrl
from sqlalchemy.orm.session import Session
from starlette import status
from starlette.status import HTTP_200_OK
from starlette.testclient import TestClient
from torch_geometric.loader import DataLoader

from mariner.core.config import settings
from mariner.entities import Dataset as DatasetEntity
from mariner.entities import Model as ModelEntity
from mariner.entities import ModelVersion
from mariner.schemas.dataset_schemas import QuantityDataType
from mariner.schemas.model_schemas import Model, ModelCreate
from mariner.stores import dataset_sql
from model_builder import layers_schema as layers
from model_builder.dataset import CustomDataset
from model_builder.model import CustomModel
from model_builder.schemas import ColumnConfig, DatasetConfig, ModelSchema
from tests.fixtures.model import mock_model, model_config, setup_create_model
from tests.fixtures.user import get_test_user
from tests.utils.utils import random_lower_string


@pytest.fixture(scope="function")
def mocked_invalid_model(some_dataset: DatasetEntity) -> ModelCreate:
    """
    Fixture of an invalid model
    """
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
                constructor_args=layers.TorchlinearConstructorArgs(
                    in_features=27, out_features=1
                ),
                forward_args=layers.TorchlinearForwardArgsReferences(input="$some"),
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


@pytest.mark.integration
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
    model_version = version["name"]
    assert version["config"]["name"] is not None
    assert "description" in version
    assert version["description"] == model.model_version_description
    assert model is not None
    db_model_config = db.query(ModelVersion).filter(
        ModelVersion.id == version["id"] and ModelVersion.name == model_version
    )
    assert db_model_config is not None


@pytest.mark.integration
def test_post_models_on_existing_model_creates_new_version(
    db: Session,
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    some_model_integration: Model,
):
    new_version = some_model_integration.versions[0].copy()
    res = client.post(
        f"{settings.API_V1_STR}/models/",
        json={
            "name": some_model_integration.name,
            "config": new_version.config.dict(),
            "modelVersionDescription": "This should be  version 2",
        },
        headers=normal_user_token_headers,
    )
    assert res.status_code == status.HTTP_200_OK
    model = (
        db.query(ModelEntity)
        .filter(ModelEntity.name == some_model_integration.name)
        .first()
    )
    assert model
    model = Model.from_orm(model)
    assert len(model.versions) == 2
    assert model.versions[-1].name


def test_post_models_dataset_not_found(
    client: TestClient, normal_user_token_headers: dict[str, str]
):
    model = mock_model(name=random_lower_string())
    datasetname = "a dataset name that is not registered"
    model.config.dataset.name = datasetname
    res = client.post(
        f"{settings.API_V1_STR}/models/",
        json=model.dict(),
        headers=normal_user_token_headers,
    )
    assert res.status_code == status.HTTP_404_NOT_FOUND
    assert res.json()["detail"] == f'Dataset "{datasetname}" not found'


@pytest.mark.integration
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


def test_get_model_options(
    client: TestClient, normal_user_token_headers: dict[str, str]
):
    res = client.get(
        f"{settings.API_V1_STR}/models/options", headers=normal_user_token_headers
    )
    assert res.status_code == HTTP_200_OK
    payload = res.json()

    def assert_component_info(component_dict: dict):
        assert "docs" in component_dict
        assert "docsLink" in component_dict
        assert "classPath" in component_dict
        assert "outputType" in component_dict
        assert "component" in component_dict
        assert "forwardArgsSummary" in component_dict["component"]
        assert "constructorArgsSummary" in component_dict["component"]
        assert isinstance(component_dict["docs"], str)
        assert AnyHttpUrl(component_dict["docs"], scheme="https") is not None

    for layer_payload in payload:
        assert_component_info(layer_payload)


def test_add_version_to_model():
    pass


def test_update_model():
    pass


@pytest.mark.integration
def test_delete_model(
    client: TestClient,
    db: Session,
    normal_user_token_headers: dict[str, str],
    some_dataset: DatasetEntity,
):
    model = setup_create_model(
        client, normal_user_token_headers, dataset_name=some_dataset.name
    )
    res = client.delete(
        f"{settings.API_V1_STR}/models/{model.id}", headers=normal_user_token_headers
    )
    assert res.status_code == 200
    assert not db.query(ModelEntity).filter(ModelEntity.id == model.id).first()


@pytest.mark.integration
def test_post_predict(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    some_trained_model: Model,
):
    model_version = some_trained_model.versions[-1].id
    route = f"{settings.API_V1_STR}/models/{model_version}/predict"
    res = client.post(
        route,
        json={
            "smiles": [
                "CCCC",
                "CCCCC",
                "CCCCCCC",
            ],
            "mwt": [3, 1, 9],
        },
        headers=normal_user_token_headers,
    )
    assert res.status_code == 200
    body = res.json()
    assert len(body) == 3


@pytest.mark.integration
def test_post_predict_fails_untrained_model(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    some_model_integration: Model,
):
    model_version = some_model_integration.versions[-1].id
    route = f"{settings.API_V1_STR}/models/{model_version}/predict"
    res = client.post(
        route,
        json={
            "smiles": [
                "CCCC",
                "CCCCC",
                "CCCCCCC",
            ],
            "mwt": [3, 1, 9],
        },
        headers=normal_user_token_headers,
    )
    assert res.status_code == status.HTTP_400_BAD_REQUEST
    body = res.json()
    assert body["detail"] == "Model version was not trained yet"


@pytest.mark.integration
def test_post_predict_validates_smiles(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    some_model_integration: Model,
):
    model_version = some_model_integration.versions[-1].id
    route = f"{settings.API_V1_STR}/models/{model_version}/predict"
    res = client.post(
        route,
        json={
            "smiles": [
                "abc",
                "abdc",
                "aebdc",
            ],
            "mwt": [0.3, 0.1, 0.9],
        },
        headers=normal_user_token_headers,
    )
    assert res.status_code == 400


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


def test_post_check_config_good_model(
    some_dataset: DatasetEntity,
    normal_user_token_headers: dict[str, str],
    client: TestClient,
):
    regressor: ModelSchema = model_config(dataset_name=some_dataset.name)
    res = client.post(
        f"{settings.API_V1_STR}/models/check-config",
        headers=normal_user_token_headers,
        json=regressor.dict(),
    )
    assert res.status_code == 200
    body = res.json()
    assert "output" in body


def test_post_check_config_good_model2(
    some_dataset: DatasetEntity,
    normal_user_token_headers: dict[str, str],
    client: TestClient,
):
    regressor: ModelSchema = model_config(
        dataset_name=some_dataset.name, model_type="regressor-with-categorical"
    )
    res = client.post(
        f"{settings.API_V1_STR}/models/check-config",
        headers=normal_user_token_headers,
        json=regressor.dict(),
    )
    assert res.status_code == 200
    body = res.json()
    assert "output" in body


def test_post_check_config_bad_model(
    db: Session,
    some_dataset: DatasetEntity,
    normal_user_token_headers: dict[str, str],
    client: TestClient,
):

    model_path = "tests/data/model_fails_on_training.yml"
    with open(model_path, "rU") as f:
        regressor: ModelSchema = ModelSchema.from_yaml(f.read())
        regressor.dataset.name = some_dataset.name

    model = CustomModel(regressor)
    dataset = dataset_sql.dataset_store.get_by_name(db, regressor.dataset.name)
    torch_dataset = CustomDataset(dataset.get_dataframe(), config=regressor)
    dataloader = DataLoader(torch_dataset, batch_size=1)
    batch = next(iter(dataloader))
    out = model(batch)
    assert out != None, "Model forward is fine"
    try:
        model.training_step(batch, 0)
        pytest.fail("training_step should fail")
    except Exception:
        pass

    res = client.post(
        f"{settings.API_V1_STR}/models/check-config",
        headers=normal_user_token_headers,
        json=regressor.dict(),
    )
    assert res.status_code == 200
    body = res.json()
    assert body["output"] == None
    assert not body["output"]
    assert "stackTrace" in body
    print(body["stackTrace"])
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
    assert len(body["name"]) >= 4
