from typing import Literal, Optional

import mlflow.tracking
import yaml
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from mariner.core.config import settings
from mariner.entities import Model as ModelEntity
from mariner.entities import ModelVersion
from mariner.schemas.model_schemas import Model, ModelCreate
from model_builder.schemas import ModelSchema
from tests.utils.utils import random_lower_string

ModelType = Literal["regressor", "regressor-with-categorical", "classifier"]


def model_config(
    model_type: ModelType = "regressor", dataset_name: Optional[str] = None
) -> ModelSchema:
    path = get_config_path_for_model_type(model_type)
    with open(path, "rb") as f:
        if path.endswith("yaml") or path.endswith("yml"):
            schema = ModelSchema.from_yaml(f.read())
        elif path.endswith("json"):
            schema = ModelSchema.parse_raw(f.read())
        else:
            raise NotImplementedError(
                f"Only know how to read json or yaml model configs, got file: {path}"
            )
        if dataset_name:
            schema.dataset.name = dataset_name
        return schema


def get_config_path_for_model_type(model_type: ModelType) -> str:
    if model_type == "regressor":
        model_path = "tests/data/small_regressor_schema.yaml"
    elif model_type == "classifier":
        model_path = "tests/data/small_classifier_schema.yaml"
    elif model_type == "regressor-with-categorical":
        model_path = "tests/data/small_regressor_schema2.json"
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
    with open(model_path, "rb") as f:
        config_dict = yaml.unsafe_load(f.read())
        config = ModelSchema(**config_dict)
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
    print("data is %r" % data)
    res = client.post(
        f"{settings.API_V1_STR}/models/",
        json=data,
        headers=headers,
    )
    print(res.json())
    assert res.status_code == status.HTTP_200_OK
    return Model.parse_obj(res.json())


def teardown_create_model(db: Session, model: Model):
    obj = db.query(ModelVersion).filter(ModelVersion.model_id == model.id).first()
    db.delete(obj)
    obj = db.query(ModelEntity).filter(ModelEntity.id == model.id).first()
    if obj:
        db.delete(obj)
        db.flush()
        mlflowclient = mlflow.tracking.MlflowClient()
        mlflowclient.delete_registered_model(model.mlflow_name)
