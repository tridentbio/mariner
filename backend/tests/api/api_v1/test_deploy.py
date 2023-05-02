from contextlib import contextmanager
from typing import Dict, Literal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm.session import Session

from mariner.core.config import settings
from mariner.entities.dataset import Dataset
from mariner.entities.deploy import ShareStrategy
from mariner.entities.model import Model
from mariner.schemas.deploy_schemas import (
    Deploy,
    DeployCreateRepo,
    PermissionCreateRepo,
)
from mariner.stores.deploy_sql import deploy_store
from tests.fixtures.deploys import mock_deploy
from tests.fixtures.model import setup_create_model_db, teardown_create_model
from tests.fixtures.user import get_random_test_user, get_test_user
from tests.utils.utils import random_lower_string


@pytest.fixture(scope="module")
def deploy(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    db: Session,
    some_model: Model,
) -> Deploy:
    """Deploy fixture. Owner: default test_user"""
    deploy_data = mock_deploy(some_model)
    response = client.post(
        f"{settings.API_V1_STR}/deploy/",
        json=deploy_data,
        headers=normal_user_token_headers,
    )
    yield Deploy(**response.json())
    db.execute("DELETE FROM deploy where id = :id", {"id": response.json()["id"]})


@contextmanager
def create_temporaly_deploy(
    db: Session,
    share_by: Literal["user", "org", "public", None],
    dataset: Dataset,
    owner: Literal["test_user", "random_user"] = "random_user",
):
    """Create a deploy temporaly and delete it after the test.
    Recomended to be used with "with" statement.

    Recommended to create deploys dynamically during the tests
    by choosing an owner other than test_user and a share mode
    to be shared with test_user.
    If you only need a static deploy to test_user, use the "deploy" fixture.

    Args:
        db (Session): SQLAlchemy session.
        share_by (str): Share mode to be shared with test_user.
        dataset (Dataset): Dataset to be used in the deploy.
        owner (str, optional): Owner of the deploy. Defaults to "random_user".

    Return:
        Deploy: Deploy created.
    """
    some_model: Model = setup_create_model_db(db, dataset, owner)
    test_user = get_test_user(db)
    deploy = deploy_store.create(
        db=db,
        obj_in=DeployCreateRepo(
            created_by_id=some_model.created_by_id,
            name=random_lower_string(),
            model_version_id=some_model.versions[0].id,
            rate_limit_value=100,
            share_strategy=(
                ShareStrategy.PUBLIC if share_by == "public" else ShareStrategy.PRIVATE
            ),
        ),
    )
    deploy = Deploy.from_orm(deploy)

    if share_by not in ["public", None]:
        permission = {
            "user": PermissionCreateRepo(
                deploy_id=deploy.id,
                user_id=test_user.id,
            ),
            "org": PermissionCreateRepo(
                deploy_id=deploy.id,
                organization=f'@{test_user.email.split("@")[1]}',
            ),
        }
        deploy_store.create_permission(db, permission[share_by])

    yield deploy

    teardown_create_model(db, some_model, skip_mlflow=True)
    db.execute("delete from deploy where id = :id", {"id": deploy.id})


@pytest.mark.parametrize("another_user_share_mode", ("user", "org", "public"))
def test_get_deploys(
    db: Session,
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    some_dataset: Dataset,
    another_user_share_mode: Literal["user", "org", "public"],
) -> None:
    """Test to route to get deploys. Should return a list of deploys.
    GET route should return all deploys that user has access and match filters.

    Deploys that user has access:
     - Deploys shared with user by other users.
     - Deploys shared with user's organization (email sufix).
     - Deploys with public mode. (only if publicMode filter is not "exclude")

    Args:
        db (Session): SQLAlchemy session.
        client (TestClient): FastAPI test client.
        normal_user_token_headers (dict[str, str]): Authorization header.
        some_dataset (Dataset): Dataset to be used in the deploy.
        another_user_share_mode (str):
            Share mode to be shared with test_user in each case test.
    """
    with create_temporaly_deploy(
        db, another_user_share_mode, some_dataset
    ) as some_deploy:
        query = None
        if another_user_share_mode == "public":
            query = {"publicMode": "only"}

        r = client.get(
            f"{settings.API_V1_STR}/deploy",
            headers=normal_user_token_headers,
            params=query,
        )
        assert r.status_code == 200
        payload = r.json()
        assert isinstance(payload["data"], list)
        assert any(
            [some_deploy.id == d["id"] for d in payload["data"]]
        ), "Should have the created deploy in the list."


def test_get_my_deploys(
    db: Session,
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    deploy: Deploy,
) -> None:
    """Test to route to get deploys with created_by_id filter.
    Should return a list of deploys created by user.

    Args:
        db (Session): SQLAlchemy session.
        client (TestClient): FastAPI test client.
        normal_user_token_headers (dict[str, str]): Authorization header.
        deploy (Deploy): Deploy fixture.
    """
    user = get_test_user(db)
    r = client.get(
        f"{settings.API_V1_STR}/deploy",
        headers=normal_user_token_headers,
        params={"created_by_id": user.id},
    )
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload["data"], list)
    assert any(
        [deploy.id == d["id"] for d in payload["data"]]
    ), "Should have the created deploy in the list."


def test_create_deploy(
    db: Session,
    client: TestClient,
    some_model: Model,
    normal_user_token_headers: Dict[str, str],
) -> None:
    """Test create deploy route. Should create a deploy in the database.

    Args:
        db (Session): SQLAlchemy session.
        client (TestClient): FastAPI test client.
        some_model (Model): Model fixture.
        normal_user_token_headers (dict[str, str]): Authorization header.
    """
    deploy_data = mock_deploy(some_model, share_strategy="public")
    response = client.post(
        f"{settings.API_V1_STR}/deploy/",
        json=deploy_data,
        headers=normal_user_token_headers,
    )
    assert response.status_code == 200
    payload = response.json()
    deploy = deploy_store.get(db, payload["id"])
    assert deploy.name == deploy_data["name"]

    db_data = deploy_store.get(db, payload["id"])
    assert bool(db_data), "Should have created the deploy in the database."
    assert db_data.name == deploy_data["name"], "Should have the same name."


def test_update_deploy(
    client: TestClient, normal_user_token_headers: Dict[str, str], deploy: Deploy
) -> None:
    """Test update deploy route. Should update the deploy in the database.

    Args:
        client (TestClient): FastAPI test client.
        normal_user_token_headers (dict[str, str]): Authorization header.
        deploy (Deploy): Deploy fixture.
    """
    updated_deploy = {"name": "Updated Name", "share_strategy": "public"}
    r = client.put(
        f"{settings.API_V1_STR}/deploy/{deploy.id}",
        json=updated_deploy,
        headers=normal_user_token_headers,
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["name"] == "Updated Name", "Should have updated the name."
    assert payload["shareStrategy"] == "public", "Should have updated to public."
    assert bool(
        payload["shareUrl"]
    ), "Should have a share url after updating to public."


def test_delete_deploy(
    db: Session,
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    some_dataset: Dataset,
) -> None:
    """Test delete deploy route. GET route should not return the deleted deploy.

    Args:
        db (Session): SQLAlchemy session.
        client (TestClient): FastAPI test client.
        normal_user_token_headers (dict[str, str]): Authorization header.
        some_dataset (Dataset): Dataset to be used in the deploy.
    """
    with create_temporaly_deploy(
        db, "public", some_dataset, "test_user"
    ) as some_deploy:
        r = client.delete(
            f"{settings.API_V1_STR}/deploy/{some_deploy.id}",
            headers=normal_user_token_headers,
        )
        assert r.status_code == 200
        payload = r.json()
        assert payload["id"] == some_deploy.id

        r = client.get(
            f"{settings.API_V1_STR}/deploy",
            headers=normal_user_token_headers,
            params={"publicMode": "only", "name": some_deploy.name},
        )
        assert r.status_code == 200
        payload = r.json()
        assert all(
            [some_deploy.id != d["id"] for d in payload["data"]]
        ), "Should not have the deploy in the list since it was deleted."


def test_create_permission(
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    db: Session,
    deploy: Deploy,
) -> None:
    """Test create permission route. Should create a permission in the database.
    GET route should return the deploy with the permission.

    Args:
        client (TestClient): FastAPI test client.
        normal_user_token_headers (dict[str, str]): Authorization header.
        db (Session): SQLAlchemy session.
        deploy (Deploy): Deploy fixture.
    """
    test_user = get_random_test_user(db)
    permission_data = PermissionCreateRepo(deploy_id=deploy.id, user_id=test_user.id)
    r = client.post(
        f"{settings.API_V1_STR}/deploy/create-permission",
        json=permission_data.dict(exclude_none=True),
        headers=normal_user_token_headers,
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["id"] == deploy.id

    r = client.get(
        f"{settings.API_V1_STR}/deploy",
        headers=normal_user_token_headers,
        params={"name": deploy.name, "created_by_id": test_user.id},
    )
    payload = r.json()

    for api_deploy in payload["data"]:
        if api_deploy["id"] == deploy.id:
            assert test_user.id in api_deploy["usersIdAllowed"]


def test_delete_permission(
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    db: Session,
    deploy: Deploy,
) -> None:
    """Test delete permission route. Should delete a permission in the database.
    GET route should not return the deploy after deleting the permission.

    Args:
        client (TestClient): FastAPI test client.
        normal_user_token_headers (dict[str, str]): Authorization header.
        db (Session): SQLAlchemy session.
        deploy (Deploy): Deploy fixture.
    """
    test_user = get_random_test_user(db)
    permission_data = PermissionCreateRepo(deploy_id=deploy.id, user_id=test_user.id)
    r = client.post(
        f"{settings.API_V1_STR}/deploy/create-permission",
        json=permission_data.dict(exclude_none=True),
        headers=normal_user_token_headers,
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["id"] == deploy.id

    r = client.post(
        f"{settings.API_V1_STR}/deploy/delete-permission",
        json=permission_data.dict(),
        headers=normal_user_token_headers,
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["id"] == deploy.id

    r = client.get(
        f"{settings.API_V1_STR}/deploy",
        headers=normal_user_token_headers,
        params={"name": deploy.name, "created_by_id": test_user.id},
    )
    payload = r.json()

    for api_deploy in payload["data"]:
        if api_deploy["id"] == deploy.id:
            assert test_user.id not in api_deploy["usersIdAllowed"]


def test_get_public_deploy(
    db: Session,
    client: TestClient,
    some_dataset: Dataset,
    normal_user_token_headers: Dict[str, str],
) -> None:
    """Test get public deploy route. Should return the deploy.
    Public deploy should be accessible by anyone by shareUrl property.

    Args:
        client (TestClient): FastAPI test client.
        normal_user_token_headers (dict[str, str]): Authorization header.
        deploy (Deploy): Deploy fixture.
    """
    with create_temporaly_deploy(db, None, some_dataset, "test_user") as some_deploy:
        # Update to public to get the share url.
        updated_deploy = {"share_strategy": "public"}
        r = client.put(
            f"{settings.API_V1_STR}/deploy/{some_deploy.id}",
            json=updated_deploy,
            headers=normal_user_token_headers,
        )
        assert r.status_code == 200
        deploy = r.json()
        public_url = deploy["shareUrl"]
        assert bool(public_url), "Should have a public url after updating to public."

        r = client.get(public_url)
        assert (
            r.status_code == 200
        ), "Should be accessible by anyone without authorization."

        payload = r.json()
        assert payload["id"] == some_deploy.id, "Should have the same id."
        assert payload["name"] == some_deploy.name, "Should have the same name."
