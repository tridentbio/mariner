from contextlib import contextmanager
from typing import Any, Dict, Literal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm.session import Session

from mariner.core.config import settings
from mariner.entities.dataset import Dataset
from mariner.entities.deployment import ShareStrategy
from mariner.entities.model import Model
from mariner.schemas.deployment_schemas import (
    Deployment,
    DeploymentCreateRepo,
    PermissionCreateRepo,
)
from mariner.stores.deployment_sql import deployment_store
from tests.fixtures.deployments import mock_deployment
from tests.fixtures.model import setup_create_model_db, teardown_create_model
from tests.fixtures.user import get_random_test_user, get_test_user
from tests.utils.utils import random_lower_string


@pytest.fixture(scope="module")
def deployment_fixture(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    some_model: Model,
) -> Deployment:
    """Deployment fixture. Owner: default test_user"""
    deployment_data = mock_deployment(some_model)
    response = client.post(
        f"{settings.API_V1_STR}/deployments/",
        json=deployment_data,
        headers=normal_user_token_headers,
    )
    return Deployment(**response.json())


@contextmanager
def create_temporary_deployment(
    db: Session,
    share_by: Literal["user", "org", "public", None],
    dataset: Dataset,
    owner: Literal["test_user", "random_user"] = "random_user",
):
    """Create a deployment temporaly and delete it after the test.
    You can set the owner and a share mode to be shared with default test_user.

    Recommendations:
        - use with "with" statement to delete the deployment after the clause ends.
        - if you just need a deployment to test_user without any sharing, use
        the "deployment" fixture instead.

    Args:
        db: SQLAlchemy session.
        share_by: Share mode to be shared with test_user.
        dataset: Dataset to be used in the deployment.
        owner: Owner of the deployment. Defaults to "random_user".

    Return:
        Deployment: Deployment created.
    """
    some_model: Model = setup_create_model_db(db, dataset, owner)
    test_user = get_test_user(db)
    deployment = deployment_store.create(
        db=db,
        obj_in=DeploymentCreateRepo(
            created_by_id=some_model.created_by_id,
            name=random_lower_string(),
            model_version_id=some_model.versions[0].id,
            prediction_rate_limit_value=100,
            share_strategy=(
                ShareStrategy.PUBLIC if share_by == "public" else ShareStrategy.PRIVATE
            ),
        ),
    )
    deployment = Deployment.from_orm(deployment)

    if share_by not in ["public", None]:
        permission = {
            "user": PermissionCreateRepo(
                deployment_id=deployment.id,
                user_id=test_user.id,
            ),
            "org": PermissionCreateRepo(
                deployment_id=deployment.id,
                organization=f'@{test_user.email.split("@")[1]}',
            ),
        }
        deployment_store.create_permission(db, permission[share_by])

    yield deployment
    teardown_create_model(db, some_model, skip_mlflow=True)


@pytest.mark.parametrize("another_user_share_mode", ("user", "org", "public"))
def test_get_deployments(
    db: Session,
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    some_dataset: Dataset,
    another_user_share_mode: Literal["user", "org", "public"],
) -> None:
    """Test to route to get deployments. Should return a list of deployments.
    GET route should return all deployments that user has access and match filters.

    Deployments that user has access:
     - Deployments shared with user by other users.
     - Deployments shared with user's organization (email sufix).
     - Deployments with public mode. (only if publicMode filter is not "exclude")

    Args:
        db: SQLAlchemy session.
        client: FastAPI test client.
        normal_user_token_headers: Authorization header.
        some_dataset: Dataset fixture to be used in the deployment.
        another_user_share_mode:
            Share mode to be shared with test_user in each case test.
    """
    with create_temporary_deployment(
        db, another_user_share_mode, some_dataset
    ) as some_deployment:
        query = None
        if another_user_share_mode == "public":
            query = {"publicMode": "only"}

        r = client.get(
            f"{settings.API_V1_STR}/deployments",
            headers=normal_user_token_headers,
            params=query,
        )
        assert r.status_code == 200
        payload = r.json()
        assert isinstance(payload["data"], list)
        assert any(
            [some_deployment.id == d["id"] for d in payload["data"]]
        ), "Should have the created deployment in the list."


def test_get_my_deployments(
    db: Session,
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    deployment_fixture: Deployment,
) -> None:
    """Test to route to get deployments with created_by_id filter.
    Should return a list of deployments created by user.

    Args:
        db: SQLAlchemy session.
        client: FastAPI test client.
        normal_user_token_headers: Authorization header.
        deployment_fixture
    """
    r = client.get(
        f"{settings.API_V1_STR}/deployments",
        headers=normal_user_token_headers,
        params={"accessMode": "owned"},
    )
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload["data"], list)
    assert any(
        [deployment_fixture.id == d["id"] for d in payload["data"]]
    ), "Should have the created deployment in the list."


def test_create_deployment(
    db: Session,
    client: TestClient,
    some_model: Model,
    normal_user_token_headers: Dict[str, str],
) -> None:
    """Test create deployment route. Should create a deployment in the database.

    Args:
        db: SQLAlchemy session.
        client: FastAPI test client.
        some_model: Model fixture.
        normal_user_token_headers: Authorization header.
    """
    deployment_data = mock_deployment(some_model, share_strategy="public")
    response = client.post(
        f"{settings.API_V1_STR}/deployments/",
        json=deployment_data,
        headers=normal_user_token_headers,
    )
    assert response.status_code == 200
    payload = response.json()
    deployment = deployment_store.get(db, payload["id"])
    assert deployment.name == deployment_data["name"]

    db_data = deployment_store.get(db, payload["id"])
    assert bool(db_data), "Should have created the deployment in the database."
    assert db_data.name == deployment_data["name"], "Should have the same name."


def test_update_deployment(
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    deployment_fixture: Deployment,
) -> None:
    """Test update deployment route. Should update the deployment in the database.

    Args:
        client: FastAPI test client.
        normal_user_token_headers: Authorization header.
        deployment_fixture.
    """
    updated_deployment = {"name": "Updated Name", "share_strategy": "public"}
    r = client.put(
        f"{settings.API_V1_STR}/deployments/{deployment_fixture.id}",
        json=updated_deployment,
        headers=normal_user_token_headers,
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["name"] == "Updated Name", "Should have updated the name."
    assert payload["shareStrategy"] == "public", "Should have updated to public."
    assert bool(
        payload["shareUrl"]
    ), "Should have a share url after updating to public."


def test_delete_deployment(
    db: Session,
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    some_dataset: Dataset,
) -> None:
    """Test delete deployment route. GET route should not return the deleted deployment.

    Args:
        db: SQLAlchemy session.
        client: FastAPI test client.
        normal_user_token_headers: Authorization header.
        some_dataset: Dataset fixture to be used in the deployment.
    """
    with create_temporary_deployment(
        db, "public", some_dataset, "test_user"
    ) as some_deployment:
        r = client.delete(
            f"{settings.API_V1_STR}/deployments/{some_deployment.id}",
            headers=normal_user_token_headers,
        )
        assert r.status_code == 200
        payload = r.json()
        assert payload["id"] == some_deployment.id

        r = client.get(
            f"{settings.API_V1_STR}/deployments",
            headers=normal_user_token_headers,
            params={"publicMode": "only", "name": some_deployment.name},
        )
        assert r.status_code == 200
        payload = r.json()
        assert all(
            [some_deployment.id != d["id"] for d in payload["data"]]
        ), "Should not have the deployment in the list since it was deleted."


def test_create_permission(
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    db: Session,
    deployment_fixture: Deployment,
) -> None:
    """Test create permission route. Should create a permission in the database.
    GET route should return the deployment with the permission.

    Args:
        client: FastAPI test client.
        normal_user_token_headers: Authorization header.
        db: SQLAlchemy session.
        deployment_fixture.
    """
    test_user = get_random_test_user(db)
    permission_data = PermissionCreateRepo(
        deployment_id=deployment_fixture.id, user_id=test_user.id
    )
    r = client.post(
        f"{settings.API_V1_STR}/deployments/create-permission",
        json=permission_data.dict(exclude_none=True),
        headers=normal_user_token_headers,
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["id"] == deployment_fixture.id

    r = client.get(
        f"{settings.API_V1_STR}/deployments",
        headers=normal_user_token_headers,
        params={"name": deployment_fixture.name, "created_by_id": test_user.id},
    )
    payload = r.json()

    for api_deployment in payload["data"]:
        if api_deployment["id"] == deployment_fixture.id:
            assert test_user.id in api_deployment["usersIdAllowed"]


def test_delete_permission(
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    db: Session,
    deployment_fixture: Deployment,
) -> None:
    """Test delete permission route. Should delete a permission in the database.
    GET route should not return the deployment after deleting the permission.

    Args:
        client: FastAPI test client.
        normal_user_token_headers: Authorization header.
        db: SQLAlchemy session.
        deployment_fixture.
    """
    test_user = get_random_test_user(db)
    permission_data = PermissionCreateRepo(
        deployment_id=deployment_fixture.id, user_id=test_user.id
    )
    r = client.post(
        f"{settings.API_V1_STR}/deployments/create-permission",
        json=permission_data.dict(exclude_none=True),
        headers=normal_user_token_headers,
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["id"] == deployment_fixture.id

    r = client.post(
        f"{settings.API_V1_STR}/deployments/delete-permission",
        json=permission_data.dict(),
        headers=normal_user_token_headers,
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["id"] == deployment_fixture.id

    r = client.get(
        f"{settings.API_V1_STR}/deployments",
        headers=normal_user_token_headers,
        params={"name": deployment_fixture.name, "accessMode": "owned"},
    )
    payload = r.json()

    for api_deployment in payload["data"]:
        if api_deployment["id"] == deployment_fixture.id:
            assert test_user.id not in api_deployment["usersIdAllowed"]


def test_get_public_deployment(
    db: Session,
    client: TestClient,
    some_dataset: Dataset,
    normal_user_token_headers: Dict[str, str],
) -> None:
    """Test get public deployment route. Should return the deployment.
    Public deployment should be accessible by anyone by shareUrl property.

    Args:
        client: FastAPI test client.
        normal_user_token_headers: Authorization header.
        db: SQLAlchemy session.
        some_dataset: Dataset fixture to be used in the deployment.
    """
    with create_temporary_deployment(
        db, None, some_dataset, "test_user"
    ) as some_deployment:
        # Update to public to get the share url.
        updated_deployment = {"share_strategy": "public"}
        r = client.put(
            f"{settings.API_V1_STR}/deployments/{some_deployment.id}",
            json=updated_deployment,
            headers=normal_user_token_headers,
        )
        assert r.status_code == 200
        deployment = r.json()
        public_url = deployment["shareUrl"]
        assert bool(public_url), "Should have a public url after updating to public."

        r = client.get(public_url)
        assert (
            r.status_code == 200
        ), "Should be accessible by anyone without authorization."

        payload = r.json()
        assert payload["id"] == some_deployment.id, "Should have the same id."
        assert payload["name"] == some_deployment.name, "Should have the same name."


@pytest.fixture(scope="module")
def predict_req_data():
    return {
        "smiles": [
            "CCCC",
            "CCCCC",
            "CCCCCCC",
        ],
        "mwt": [3, 1, 9],
    }


def test_post_make_prediction(
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    some_deployment: Deployment,
    predict_req_data: Dict[str, Any],
):
    r = client.post(
        f"{settings.API_V1_STR}/deployments/{some_deployment.id}/predict",
        json=predict_req_data,
        headers=normal_user_token_headers,
    )
    assert (
        r.status_code == 404
    ), "Should not find the deployment instance on ray since it's not running."

    r = client.put(
        f"{settings.API_V1_STR}/deployments/{some_deployment.id}",
        json={
            "share_strategy": "public",
            "prediction_rate_limit_value": 1,
            "prediction_rate_limit_unit": "day",
            "status": "active",
        },
        headers=normal_user_token_headers,
    )
    assert r.status_code == 200, "Should update deployment instance status to active."

    r = client.post(
        f"{settings.API_V1_STR}/deployments/{some_deployment.id}/predict",
        json=predict_req_data,
        headers=normal_user_token_headers,
    )
    assert r.status_code == 200
    payload = r.json()
    assert "tpsa" in payload, "'tpsa' column should be in prediction"
    assert isinstance(payload["tpsa"], list), "'tpsa' column should be a list"

    r = client.post(
        f"{settings.API_V1_STR}/deployments/{some_deployment.id}/predict",
        json=predict_req_data,
        headers=normal_user_token_headers,
    )
    assert (
        r.status_code == 429
    ), "Should return 429 status code when rate limit is exceeded"
