from typing import Dict

from fastapi.testclient import TestClient

from mariner.core.config import settings


def test_get_users_normal_user_me(
    client: TestClient, normal_user_token_headers: Dict[str, str]
) -> None:
    r = client.get(f"{settings.API_V1_STR}/users/me", headers=normal_user_token_headers)
    current_user = r.json()
    assert current_user
    assert current_user["isActive"] is True
    assert current_user["isSuperuser"] is False
    assert current_user["email"] == settings.EMAIL_TEST_USER
