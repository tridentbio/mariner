"""
Tests the mariner.models package
"""

import time
from typing import Literal

import pytest
import ray
from pydantic import AnyHttpUrl
from sqlalchemy.orm.session import Session
from starlette import status
from starlette.status import HTTP_200_OK
from starlette.testclient import TestClient

from fleet.base_schemas import TorchModelSpec
from fleet.dataset_schemas import (
    ColumnConfig,
    TargetTorchColumnConfig,
    TorchDatasetConfig,
)
from fleet.model_builder import layers_schema as layers
from fleet.model_builder.schemas import TorchModelSchema
from fleet.ray_actors.model_check_actor import TrainingCheckResponse
from fleet.ray_actors.tasks import get_task_control
from mariner.core.config import get_app_settings
from mariner.db.session import SessionLocal
from mariner.entities import Dataset as DatasetEntity
from mariner.entities import Model as ModelEntity
from mariner.entities import ModelVersion
from mariner.schemas.dataset_schemas import QuantityDataType
from mariner.schemas.model_schemas import Model, ModelCreate
from tests.fixtures.model import mock_model, setup_create_model
from tests.fixtures.user import get_test_user
from tests.utils.utils import random_lower_string


@pytest.fixture(scope="function")
def mocked_invalid_model(some_dataset: DatasetEntity) -> ModelCreate:
    """
    Fixture of an invalid model
    """
    config = TorchModelSpec(
        name=random_lower_string(),
        dataset=TorchDatasetConfig(
            name=some_dataset.name,
            featurizers=[],
            feature_columns=[
                ColumnConfig(
                    name="mwt",
                    data_type=QuantityDataType(
                        domain_kind="numeric", unit="mole"
                    ),
                )
            ],
            target_columns=[
                TargetTorchColumnConfig(
                    name="tpsa",
                    data_type=QuantityDataType(
                        domain_kind="numeric", unit="mole"
                    ),
                    out_module="1",
                )
            ],
        ),
        spec=TorchModelSchema(
            layers=[
                layers.TorchlinearLayerConfig(
                    type="torch.nn.Linear",
                    constructor_args=layers.TorchlinearConstructorArgs(
                        in_features=27, out_features=1
                    ),
                    forward_args=layers.TorchlinearForwardArgsReferences(
                        input="$some"
                    ),
                    name="1",
                )
            ]
        ),
    )
    model = ModelCreate(
        name=config.name,
        model_description=random_lower_string(),
        model_version_description=random_lower_string(),
        config=config,
    )
    return model


# @pytest.mark.parametrize("framework", ("torch", "sklearn"))
@pytest.mark.integration
@pytest.mark.asyncio
async def test_post_models_success(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    framework: Literal["torch", "sklearn"],
    some_dataset: DatasetEntity,  # noqa
    sampl_dataset: DatasetEntity,  # noqa
):
    model = mock_model(framework=framework)
    res = client.post(
        f"{get_app_settings('server').host}/api/v1/models/",
        json=model.dict(),
        headers=normal_user_token_headers,
    )
    body = res.json()
    assert res.status_code == HTTP_200_OK, body["detail"]
    assert body["name"] == model.name
    assert "columns" in body
    assert body["createdById"] == some_dataset.created_by_id
    assert body["description"] == model.model_description
    assert "versions" in body
    assert len(body["versions"]) == 1
    version = body["versions"][0]
    model_version = version["name"]
    assert version["config"]["name"] is not None
    assert "description" in version
    assert version["description"] == model.model_version_description
    if framework == "torch":
        assert version["checkStatus"] is None
    else:
        assert version["checkStatus"] == "OK"
    assert version["checkStackTrace"] is None
    assert model is not None

    if framework == "torch":
        # Assert the model checking task is running:
        ids, tasks = get_task_control().get_tasks(
            {"model_version_id": version["id"]}
        )
        assert (
            len(tasks) == len(ids) == 1
        ), f"Expected 1 task got {len(tasks)} tasks and {len(ids)} metadatas"
        task = tasks[0]
        result = ray.get(task)
        # Quick sleep because we're waiting for the ray task
        # not the asyncio task that wraps the ray task
        # and follows up with the model version update
        time.sleep(2)
        assert isinstance(result, TrainingCheckResponse), (
            f"Expected TrainingCheckResponse got {type(result)} "
            f"with value {result}"
        )
        print(repr(result))
        with SessionLocal() as db:
            model_version = (
                db.query(ModelVersion)
                .filter(ModelVersion.id == version["id"])
                .first()
            )
            print(model_version.id)
            assert (
                model_version.check_status == "OK"
            ), f"Expected check_status to be OK got {model_version.check_status}\nmodel_version.check_stack_trace:\n{model_version.check_stack_trace}"


@pytest.mark.integration
def test_post_models_on_existing_model_creates_new_version(
    db: Session,
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    some_model_integration: Model,
):
    new_version = some_model_integration.versions[0].copy()
    res = client.post(
        f"{get_app_settings('server').host}/api/v1/models/",
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
        f"{get_app_settings('server').host}/api/v1/models/",
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
        f"{get_app_settings('server').host}/api/v1/models/",
        headers=normal_user_token_headers,
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
        f"{get_app_settings('server').host}/api/v1/models/options",
        headers=normal_user_token_headers,
    )
    assert res.status_code == HTTP_200_OK
    payload = res.json()

    def assert_component_info(component_dict: dict):
        assert "docs" in component_dict, payload
        assert "classPath" in component_dict, payload
        assert "outputType" in component_dict, payload
        assert "component" in component_dict, payload

        assert isinstance(component_dict["docs"], str), payload
        assert (
            AnyHttpUrl(component_dict["docs"], scheme="https") is not None
        ), payload
        if component_dict["classPath"] in [
            "molfeat.trans.fp.FPVecFilteredTransformer",
            "sklearn.preprocessing.LabelEncoder",
            "sklearn.preprocessing.OneHotEncoder",
            "sklearn.preprocessing.StandardScaler",
        ]:
            return
        assert "forwardArgsSummary" in component_dict["component"], payload
        assert "constructorArgsSummary" in component_dict["component"], payload

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
        f"{get_app_settings('server').host}/api/v1/models/{model.id}",
        headers=normal_user_token_headers,
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
    route = f"{get_app_settings('server').host}/api/v1/models/{model_version}/predict"
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
    assert "tpsa" in body
    assert len(body["tpsa"]) == 3


@pytest.mark.integration
def test_post_predict_fails_untrained_model(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    some_model_integration: Model,
):
    model_version = some_model_integration.versions[-1].id
    route = f"{get_app_settings('server').host}/api/v1/models/{model_version}/predict"
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
    route = f"{get_app_settings('server').host}/api/v1/models/{model_version}/predict"
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
    client: TestClient,
    some_model: Model,
    normal_user_token_headers: dict[str, str],
):
    res = client.get(
        f"{get_app_settings('server').host}/api/v1/models/{some_model.id}/",
        headers=normal_user_token_headers,
    )
    assert res.status_code == 200
    body = res.json()
    assert body["name"] == some_model.name


def test_get_name_suggestion(
    client: TestClient, normal_user_token_headers: dict[str, str]
):
    res = client.get(
        f"{get_app_settings('server').host}/api/v1/models/name-suggestion",
        headers=normal_user_token_headers,
    )
    assert res.status_code == 200
    body = res.json()
    assert "name" in body
    assert type(body["name"]) == str
    assert len(body["name"]) >= 4
