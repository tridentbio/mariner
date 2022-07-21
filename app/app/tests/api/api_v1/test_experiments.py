from starlette.status import HTTP_200_OK
from starlette.testclient import TestClient

from app.core.config import settings
from app.features.model.schema.model import Model
from app.tests.utils.utils import random_lower_string


def test_post_experiments(
    client: TestClient,
    some_model: Model,
    normal_user_token_headers
):
    experiment_name = random_lower_string()
    version = some_model.versions[-1].model_version
    json = {
        "experimentName": experiment_name,
        "learningRate": 0.05,
        "epochs": 1,
        "modelVersion": version,
        "modelName": some_model.name,
    }
    res = client.post(
        f"{settings.API_V1_STR}/experiments/",
        json=json,
        headers=normal_user_token_headers,
    )
    print(res.json())
    assert res.status_code == HTTP_200_OK


