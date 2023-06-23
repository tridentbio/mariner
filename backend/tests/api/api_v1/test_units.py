from typing import Dict, List, Tuple

import pytest
from fastapi import status, testclient

from mariner.core.config import get_app_settings


def test_get_units(
    client: testclient.TestClient, normal_user_token_headers: Dict[str, str]
):
    res = client.get(
        f"{get_app_settings().API_V1_STR}/units?q=m",
        headers=normal_user_token_headers,
    )
    assert res.status_code == status.HTTP_200_OK
    body = res.json()
    assert {
        "name": "mole",
        "latex": "$\\mathrm{mole}$",
    } in body, "mol unit is not in the response"


units: List[Tuple[str, bool]] = [
    ("m", True),
    ("m/s", True),
    ("log(m/s)", True),
    ("ln(log(m))", True),
    ("log2(mole)/m", True),
    ("log8(cm)", True),
    ("log2(mole/m", False),
    ("ln(m/s", False),
    ("foo", False),
    ("ln(foo)", False),
]


@pytest.mark.parametrize("unit,is_valid", units)
def test_get_is_unit_valid(
    client: testclient.TestClient,
    normal_user_token_headers: Dict[str, str],
    unit: str,
    is_valid: bool,
):
    res = client.get(
        f"{get_app_settings().API_V1_STR}/units/valid?q={unit}",
        headers=normal_user_token_headers,
    )
    assert bool(res.status_code == status.HTTP_200_OK) == is_valid
    assert bool(res.content) == is_valid
