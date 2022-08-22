from typing import Dict, Optional

import mlflow
import pytest
import yaml
from sqlalchemy.orm.session import Session
from starlette import status
from starlette.testclient import TestClient

from app.core.config import settings
from app.features.dataset.model import Dataset
from app.features.model.model import Model as ModelEntity
from app.features.model.model import ModelVersion
from app.features.model.schema.configs import ModelConfig
from app.features.model.schema.model import Model, ModelCreate
from app.tests.utils.utils import random_lower_string


def mock_model(name=None, dataset_name=None) -> ModelCreate:
    model_path = "app/tests/data/test_model_hard.yaml"
    with open(model_path, "rb") as f:
        config_dict = yaml.unsafe_load(f.read())
        config = ModelConfig.parse_obj(config_dict)
        if dataset_name:
            config.dataset.name = dataset_name
        model = ModelCreate(
            name=name if name is not None else random_lower_string(),
            model_description=random_lower_string(),
            model_version_description=random_lower_string(),
            config=config,
        )
        return model


def setup_create_model(client: TestClient, headers, dataset: Optional[Dataset] = None):
    model = None
    if dataset:
        model = mock_model(dataset_name=dataset.name)
    else:
        model = mock_model()
    data = model.dict()
    res = client.post(
        f"{settings.API_V1_STR}/models/",
        json=data,
        headers=headers,
    )
    assert res.status_code == status.HTTP_200_OK
    return Model.parse_obj(res.json())


def teardown_create_model(db: Session, model: Model):
    obj = db.query(ModelVersion).filter(ModelVersion.model_id == model.id).first()
    db.delete(obj)
    obj = db.query(ModelEntity).filter(ModelEntity.id == model.id).first()
    db.delete(obj)
    db.commit()
    mlflowclient = mlflow.tracking.MlflowClient()
    mlflowclient.delete_registered_model(model.mlflow_name)


@pytest.fixture(scope="function")
def model(
    db: Session,
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    some_dataset: Dataset,
):
    db.query(ModelEntity).delete()
    db.commit()
    model = setup_create_model(client, normal_user_token_headers, some_dataset)
    return model
