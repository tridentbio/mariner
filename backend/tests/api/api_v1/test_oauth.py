"""Tests the OAuth related endpoints"""


from fastapi.testclient import TestClient


def test_get_oauth_providers(client: TestClient):
    """Tests the method to get configured oauth providers"""
    response = client.get("/api/v1/oauth-providers")
    assert response.status_code == 200
    for settings in response.json():
        assert "client_id" not in settings
        assert "client_secret" not in settings
        assert "authorization_url" not in settings
        assert "allowed_github_emails" not in settings
        assert "scope" not in settings
        assert "methods" not in settings
        assert "id" in settings
