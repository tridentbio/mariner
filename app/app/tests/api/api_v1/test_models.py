import mlflow.pyfunc
from sqlalchemy.orm.session import Session
from starlette.status import HTTP_200_OK
from starlette.testclient import TestClient

from app.core.config import settings
from app.features.user.crud import repo as user_repo


def test_upload_pytorch_serialized_model(
    db: Session,
    client: TestClient,
    normal_user_token_headers: dict[str, str],
):
    model_path = "app/tests/data/model.torch"
    user = user_repo.get_by_email(db, email=settings.EMAIL_TEST_USER)
    model_name = "Test model"
    assert user is not None
    with open(model_path, "rb") as f:
        res = client.post(
            f"{settings.API_V1_STR}/models/",
            data={
                "modelRegistryName": model_name,
            },
            files={"file": ("model.torch", f.read())},
            headers=normal_user_token_headers,
        )
        body = res.json()
        assert res.status_code == HTTP_200_OK
        assert body["name"] is str
        assert body["createdById"] == user.id
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
    body = res.json()
    models = body["data"]
    assert len(models) > 0
    # TODO: test model properties


def test_model_deploy():
    pass
