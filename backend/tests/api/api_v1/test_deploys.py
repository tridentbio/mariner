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
from tests.fixtures.model import setup_create_model_db
from tests.fixtures.user import get_test_user
from tests.utils.utils import random_lower_string


@pytest.fixture(scope="module")
def deploy(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    db: Session,
    some_model: Model,
) -> Deploy:
    deploy_data = mock_deploy(some_model)
    response = client.post(
        f"{settings.API_V1_STR}/deploy/",
        json=deploy_data,
        headers=normal_user_token_headers,
    )
    return Deploy(**response.json())


def create_another_user_deploy(
    db: Session,
    share_by: Literal["user", "org", "public"],
    dataset: Dataset,
):
    some_model: Model = setup_create_model_db(db, dataset, owner="random_user")
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

    if share_by != "public":
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


@pytest.mark.parametrize("another_user_share_mode", ("user", "org", "public"))
def test_get_deploys(
    db: Session,
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    deploy: Deploy,
    some_dataset: Dataset,
    another_user_share_mode: Literal["user", "org", "public"],
) -> None:
    some_deploy = create_another_user_deploy(db, another_user_share_mode, some_dataset)

    query = ""
    if another_user_share_mode == "public":
        query += "?public_mode=include"

    r = client.get(
        f"{settings.API_V1_STR}/deploy{query}", headers=normal_user_token_headers
    )
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload["data"], list)
    assert any(
        [deploy.id == d["id"] for d in payload["data"]]
    ), "Should have the created deploy in the list."
    assert any(
        [some_deploy.id == d["id"] for d in payload["data"]]
    ), "Should have the created deploy in the list."


def test_create_deploy(
    db: Session,
    client: TestClient,
    some_model: Model,
    normal_user_token_headers: Dict[str, str],
) -> None:
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


def test_update_deploy(
    client: TestClient, normal_user_token_headers: Dict[str, str], deploy: Deploy
) -> None:
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


# Test for delete_deploy
def test_delete_deploy(
    client: TestClient, normal_user_token_headers: Dict[str, str], deploy: Deploy
) -> None:
    # Delete the created test deploy
    r = client.delete(
        f"{settings.API_V1_STR}/deploy/{deploy.id}", headers=normal_user_token_headers
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["id"] == deploy.id


# Test for create_permission
def test_create_permission(
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    db: Session,
    deploy: Deploy,
) -> None:
    # Create a deploy to test
    user = get_test_user(db)

    # Create a permission
    permission_data = PermissionCreateRepo(deploy_id=deploy.id, user_id=user.id)
    r = client.post(
        f"{settings.API_V1_STR}/deploy/permissions",
        json=permission_data.dict(),
        headers=normal_user_token_headers,
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["id"] == deploy.id


# Test for delete_permission
def test_delete_permission(
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    deploy: Deploy,
    db: Session,
) -> None:
    user = get_test_user(db)

    # Create a permission
    permission_data = PermissionCreateRepo(deploy_id=deploy.id, user_id=user.id)
    r = client.post(
        f"{settings.API_V1_STR}/deploy/permissions",
        json=permission_data.dict(),
        headers=normal_user_token_headers,
    )
    permission_id = r.json()["id"]

    # Delete the created permission
    r = client.delete(
        f"{settings.API_V1_STR}/deploy/permissions/{permission_id}",
        headers=normal_user_token_headers,
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["id"] == deploy.id
