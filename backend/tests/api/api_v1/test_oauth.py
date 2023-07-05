"""Tests the OAuth related endpoints"""


from fastapi.testclient import TestClient
from mockito import when

import mariner.oauth


def test_get_oauth_providers(client: TestClient):
    """Tests the method to get configured oauth providers"""
    response = client.get("/api/v1/oauth-providers")

    mocked_oauth_manager = mariner.oauth.OAuthManager(
        auth_providers={
            "github": {
                "authorization_url": "",
                "client_id": "",
                "client_secret": "",
                "scope": "",
            }
        }
    )

    with when(mariner.oauth.oauth_manager).mocked_oauth_manager.thenReturn(
        mocked_oauth_manager
    ):
        assert response.status_code == 200
        assert response.json() == mocked_oauth_manager.auth_providers
