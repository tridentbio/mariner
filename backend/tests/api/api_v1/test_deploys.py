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
    test_user = get_random_test_user(db)
    permission_data = PermissionCreateRepo(deploy_id=deploy.id, user_id=test_user.id)
    r = client.post(
        f"{settings.API_V1_STR}/deploy/permissions",
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
    test_user = get_random_test_user(db)
    permission_data = PermissionCreateRepo(deploy_id=deploy.id, user_id=test_user.id)
    r = client.post(
        f"{settings.API_V1_STR}/deploy/permissions",
        json=permission_data.dict(exclude_none=True),
        headers=normal_user_token_headers,
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["id"] == deploy.id

    r = client.delete(
        f"{settings.API_V1_STR}/deploy/permissions",
        params=permission_data.dict(),
        headers=normal_user_token_headers,
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["id"] == deploy.id
