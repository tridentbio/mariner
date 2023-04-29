import json
from typing import Dict

import pytest
from fastapi.testclient import TestClient
from mariner.schemas.deploy_schemas import Deploy, PermissionCreateRepo, DeployUpdateRepo
from sqlalchemy.orm.session import Session
from starlette import status

from mariner.core.config import settings
from mariner.stores.deploy_sql import deploy_store
from tests.fixtures.deploys import mock_deploy
from mariner.models import Model
from tests.fixtures.user import get_test_user

@pytest.fixture(scope="module")
def deploy(
    client: TestClient, 
    normal_user_token_headers: dict[str, str], 
    db: Session, 
    some_model: Model
) -> Deploy:
    deploy_data = mock_deploy(some_model)
    response = client.post(
        f"{settings.API_V1_STR}/deploy/", 
        json=deploy_data, 
        headers=normal_user_token_headers
    )
    return Deploy(**response.json())

def test_get_all_deploys(
    client: TestClient, 
    normal_user_token_headers: Dict[str, str],
    deploy: Deploy
) -> None:
    r = client.get(f"{settings.API_V1_STR}/deploy/feed", headers=normal_user_token_headers)
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload["total"], int)
    assert isinstance(payload["data"], list)
    assert any([deploy["id"] == d["id"] for d in payload["data"]])

def test_get_my_deploys(
    client: TestClient, 
    normal_user_token_headers: Dict[str, str],
    deploy: Deploy
) -> None:
    r = client.get(f"{settings.API_V1_STR}/deploy/", headers=normal_user_token_headers)
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload["total"], int)
    assert isinstance(payload["data"], list)

def test_get_my_deploy(
    client: TestClient, 
    normal_user_token_headers: Dict[str, str], 
    deploy: Deploy
) -> None:
    r = client.get(f"{settings.API_V1_STR}/deploy/{deploy.id}", headers=normal_user_token_headers)
    assert r.status_code == 200
    payload = r.json()
    assert payload["id"] == deploy.id

def test_create_deploy(
    db: Session,
    client: TestClient, 
    some_model: Model,
    normal_user_token_headers: Dict[str, str],
) -> None:
    deploy_data = mock_deploy(some_model, share_strategy='public')
    response = client.post(
        f"{settings.API_V1_STR}/deploy/", 
        json=deploy_data, 
        headers=normal_user_token_headers
    )
    assert response.status_code == 200
    payload = response.json()
    deploy = deploy_store.get(db, payload["id"])
    assert deploy.name == deploy_data["name"]

def test_update_deploy(
    client: TestClient, 
    normal_user_token_headers: Dict[str, str], 
    deploy: Deploy
) -> None:
    updated_deploy = {
        "name": "Updated Name",
        "share_strategy": "public"
    }
    r = client.put(f"{settings.API_V1_STR}/deploy/{deploy.id}", json=updated_deploy, headers=normal_user_token_headers)
    assert r.status_code == 200
    payload = r.json()
    assert payload["name"] == "Updated Name", 'Should have updated the name.'
    assert payload["share_strategy"] == "public", 'Should have updated to public.'
    assert bool(payload["share_url"]), 'Should have a share url after updating to public.'

# Test for delete_deploy
def test_delete_deploy(
    client: TestClient, 
    normal_user_token_headers: Dict[str, str], 
    deploy: Deploy
) -> None:
    # Delete the created test deploy
    r = client.delete(f"{settings.API_V1_STR}/deploy/{deploy.id}", headers=normal_user_token_headers)
    assert r.status_code == 200
    payload = r.json()
    assert payload["id"] == deploy.id

# Test for create_permission
def test_create_permission(
    client: TestClient, 
    normal_user_token_headers: Dict[str, str], 
    db: Session,
    deploy: Deploy
) -> None:
    # Create a deploy to test
    user = get_test_user(db)

    # Create a permission
    permission_data = PermissionCreateRepo(deploy_id=deploy.id, user_id=user.id)
    r = client.post(f"{settings.API_V1_STR}/deploy/permissions", json=permission_data.dict(), headers=normal_user_token_headers)
    assert r.status_code == 200
    payload = r.json()
    assert payload["id"] == deploy.id

# Test for delete_permission
def test_delete_permission(
    client: TestClient, 
    normal_user_token_headers: Dict[str, str],
    deploy: Deploy,
    db: Session
) -> None:
    user = get_test_user(db)

    # Create a permission
    permission_data = PermissionCreateRepo(deploy_id=deploy.id, user_id=user.id)
    r = client.post(f"{settings.API_V1_STR}/deploy/permissions", json=permission_data.dict(), headers=normal_user_token_headers)
    permission_id = r.json()["id"]

    # Delete the created permission
    r = client.delete(f"{settings.API_V1_STR}/deploy/permissions/{permission_id}", headers=normal_user_token_headers)
    assert r.status_code == 200
    payload = r.json()
    assert payload["id"] == deploy.id