from typing import Literal, Optional

import mlflow.tracking
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from fleet.base_schemas import TorchModelSpec
from mariner.core.config import settings
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
    model_type: ModelType = "regressor", dataset_name: Optional[str] = None
) -> TorchModelSpec:
    path = get_config_path_for_model_type(model_type)
    if path.endswith("yaml") or path.endswith("yml"):
        model = TorchModelSpec.from_yaml(path)
    else:
        model = TorchModelSpec.parse_file(path)

    # override test dataset name
    if dataset_name:
        model.dataset.name = dataset_name
    return model


def get_config_path_for_model_type(model_type: ModelType) -> str:
    if model_type == "regressor":
        model_path = "tests/data/yaml/small_regressor_schema.yaml"
    elif model_type == "classifier":
        model_path = "tests/data/yaml/small_classifier_schema.yaml"
    elif model_type == "regressor-with-categorical":
        model_path = "tests/data/json/small_regressor_schema2.json"
    else:
        raise NotImplementedError(f"No model config yaml for model type {model_type}")
    return model_path


def mock_model(
    *,
    name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    model_type: ModelType = "regressor",
) -> ModelCreate:
    model_path = get_config_path_for_model_type(model_type)
    config = TorchModelSpec.from_yaml(model_path)
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
        f"{settings.API_V1_STR}/models/",
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
    user = get_test_user(db) if owner == "test_user" else get_random_test_user(db)
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
    obj = db.query(ModelVersion).filter(ModelVersion.model_id == model.id).first()
    db.delete(obj)
    obj = db.query(ModelEntity).filter(ModelEntity.id == model.id).first()
    if obj:
        db.delete(obj)
        db.flush()
        if not skip_mlflow:
            mlflowclient = mlflow.tracking.MlflowClient()
            mlflowclient.delete_registered_model(model.mlflow_name)
