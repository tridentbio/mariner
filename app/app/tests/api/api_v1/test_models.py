import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from sqlalchemy.orm.session import Session
from starlette.status import HTTP_200_OK
from starlette.testclient import TestClient

from app.core.config import settings
from app.features.user.crud import repo as user_repo
from app.features.model.crud import repo as model_repo
from app.tests.utils.utils import random_lower_string


def test_post_models_success(
    db: Session,
    client: TestClient,
    normal_user_token_headers: dict[str, str],
):
    model_path = "app/tests/data/model.pt"
    user = user_repo.get_by_email(db, email=settings.EMAIL_TEST_USER)
    model_name: str = random_lower_string()
    model_description = "test description"
    model_version_description = "test version description"
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
    client: TestClient, normal_user_token_headers: dict[str, str]
):
    res = client.get(
        f"{settings.API_V1_STR}/models/", headers=normal_user_token_headers
    )
    assert res.status_code == HTTP_200_OK
    user = user_repo.get_by_email(db, email=settings.EMAIL_TEST_USER)
    body = res.json()
    models = body["data"]

    assert len(models) > 0
    for model in models:
        assert model.created_by_id == user.id


def test_model_deploy():
    raise NotImplemented()


def test_add_version_to_model():
    raise NotImplemented()


def test_update_model():
    raise NotImplemented()


def test_delete_model():
    raise NotImplemented()

