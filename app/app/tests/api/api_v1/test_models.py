import io
import mlflow.pyfunc

from fastapi.datastructures import UploadFile
from sqlalchemy.orm.session import Session
from starlette.status import HTTP_200_OK
from starlette.testclient import TestClient

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


def test_get_models_in_registry(
    client: TestClient, normal_user_token_headers: dict[str, str],
    db: Session
):
    res = client.get(
        f"{settings.API_V1_STR}/models/", headers=normal_user_token_headers
    )
    assert res.status_code == HTTP_200_OK
    user = get_test_user(db)
    body = res.json()
    models = body["data"]

    assert len(models) > 0
    for model in models:
        assert model.created_by_id == user.id


def test_post_models_deployment(db: Session, client: TestClient, normal_user_token_headers: dict[str, str]):
    user = get_test_user(db)
    model = mock_model(user)
    model_path = "app/tests/data/model.pt"
    with open(model_path, 'rb') as f:
        res = client.post(
            f"{settings.API_V1_STR}/models/",
            data={
                "name": model.name,
                "description": model.model_description,
                "versionDescription": model.model_version_description,
            },
            files={"file": ("model.pt", f)},
            headers=normal_user_token_headers,
        )
        assert res.status_code == HTTP_200_OK
        model = res.json()
        data = {
            'name': random_lower_string(),
            'model_name': model['name'],
            'model_version': int(model['latestVersions'][-1]['version']),
        }
        print(data)
        res = client.post(
            f"{settings.API_V1_STR}/deployments/",
            json=data,
            headers=normal_user_token_headers
        )
        assert res.status_code == HTTP_200_OK


def test_add_version_to_model():
   pass


def test_update_model():
    pass


def test_delete_model():
    pass


