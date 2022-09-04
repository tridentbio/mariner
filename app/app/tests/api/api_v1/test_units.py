from typing import Dict

from fastapi import status, testclient

from app.core.config import settings


def test_get_units(
    client: testclient.TestClient, normal_user_token_headers: Dict[str, str]
):
    res = client.get(
        f"{settings.API_V1_STR}/datasets/units?q=m",
        headers=normal_user_token_headers,
    )
    assert res.status_code == status.HTTP_200_OK
    body = res.json()
    assert {
        "name": "mole",
        "latex": "$\\mathrm{mole}$",
    } in body, "mol unit is not in the response"
