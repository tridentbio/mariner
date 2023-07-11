from typing import Literal, Optional

import mlflow.tracking
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from fleet.base_schemas import SklearnModelSpec, TorchModelSpec
from mariner.core.config import get_app_settings
from mariner.entities import Model as ModelEntity
from mariner.entities import ModelVersion
from mariner.entities.dataset import Dataset
from mariner.entities.model import ModelFeaturesAndTarget
from mariner.schemas.model_schemas import (
    Model,
    ModelCreate,
    ModelCreateRepo,
    ModelVersionCreateRepo,
)
from mariner.stores import model_sql
from tests.fixtures.user import get_random_test_user, get_test_user
from tests.utils.utils import random_lower_string

ModelType = Literal["regressor", "regressor-with-categorical", "classifier"]


def model_config(
    model_type: ModelType = "regressor",
    framework: Literal["sklearn", "torch"] = "torch",
    dataset_name: Optional[str] = None,
) -> TorchModelSpec:
    """[TODO:summary]

    Mocks a model spec parsing it from data/yaml folder.

    Args:
        model_type: the model type wanted.
        framework: Either "torch" or "sklearn"
        dataset_name: The name of the dataset used by this model.

    Returns:
        [TODO:description]
    """
    path = get_config_path_for_model_type(model_type, framework)
    if path.endswith("yaml") or path.endswith("yml"):
        model = TorchModelSpec.from_yaml(path)
    else:
        model = TorchModelSpec.parse_file(path)

    # override test dataset name
    if dataset_name:
        model.dataset.name = dataset_name
    return model


def get_config_path_for_model_type(
    model_type: ModelType, framework: Literal["sklearn", "torch"] = "torch"
) -> str:
    torch_models = {
        "regressor": "tests/data/yaml/small_regressor_schema.yaml",
        "classifier": "tests/data/yaml/small_classifier_schema.yaml",
        "regressor-with-categorical": "tests/data/json/small_regressor_schema2.json",
    }
    sklearn_models = {
        "regressor": "tests/data/yaml/sklearn_sampl_random_forest_regressor.yaml",
        "classifier": "tests/data/yaml/sklearn_hiv_random_forest_classifier.yaml",
    }

    try:
        if framework == "sklearn":
            return sklearn_models[model_type]
        elif framework == "torch":
            return torch_models[model_type]
        else:
            raise NotImplementedError(
                f"No test yamls for framework {framework}"
            )
    except KeyError as exc:
        raise NotImplementedError(
            f"No test yamls of type {model_type} found for framework {framework}"
        ) from exc


def mock_model(
    *,
    name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    model_type: ModelType = "regressor",
    framework: Literal["sklearn", "torch"] = "torch",
) -> ModelCreate:
    model_path = get_config_path_for_model_type(
        model_type, framework=framework
    )

    if framework == "torch":
        config = TorchModelSpec.from_yaml(model_path)
    elif framework == "sklearn":
        config = SklearnModelSpec.from_yaml(model_path)

    if dataset_name:
        config.dataset.name = dataset_name
    model = ModelCreate(
        name=name if name is not None else random_lower_string(),
        model_description=random_lower_string(),
        model_version_description=random_lower_string(),
        config=config,
    )
    return model


def setup_create_model(
    client: TestClient, headers: dict[str, str], **mock_model_kwargs
):
    model = mock_model(**mock_model_kwargs)
    data = model.dict(by_alias=True)
    res = client.post(
        f"{get_app_settings().API_V1_STR}/models/",
        json=data,
        headers=headers,
    )
    assert res.status_code == status.HTTP_200_OK
    return Model.parse_obj(res.json())


def setup_create_model_db(
    db: Session,
    dataset: Dataset,
    owner: Literal["test_user", "random_user"] = "test_user",
    **mock_model_kwargs,
):
    model = mock_model(**mock_model_kwargs, dataset_name=dataset.name)
    user = (
        get_test_user(db) if owner == "test_user" else get_random_test_user(db)
    )
    model_create = ModelCreateRepo(
        dataset_id=dataset.id,
        name=model.name,
        mlflow_name=random_lower_string(),
        created_by_id=user.id,
        columns=[
            ModelFeaturesAndTarget(
                column_name=feature_col.name,
                column_type="feature",
            )
            for feature_col in model.config.dataset.feature_columns
        ]
        + [
            ModelFeaturesAndTarget(
                column_name=target_col.name,
                column_type="target",
            )
            for target_col in model.config.dataset.target_columns
        ],
    )

    created_model = model_sql.model_store.create(db, model_create)
    version_create = ModelVersionCreateRepo(
        mlflow_version="1",
        mlflow_model_name=model_create.mlflow_name,
        model_id=created_model.id,
        name=model.config.name,
        config=model.config,
        description=model.model_version_description,
    )
    model_sql.model_store.create_model_version(db, version_create)
    model = db.query(ModelEntity).get(created_model.id)
    return Model.from_orm(model)


def teardown_create_model(db: Session, model: Model, skip_mlflow=False):
    obj = (
        db.query(ModelVersion)
        .filter(ModelVersion.model_id == model.id)
        .first()
    )
    db.delete(obj)
    obj = db.query(ModelEntity).filter(ModelEntity.id == model.id).first()
    if obj:
        db.delete(obj)
        db.flush()
        if not skip_mlflow:
            mlflowclient = mlflow.tracking.MlflowClient()
            mlflowclient.delete_registered_model(model.mlflow_name)
