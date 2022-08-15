import boto3
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.core.aws import Bucket


@pytest.fixture(scope="module")
def s3_existing_key() -> str:
    s3 = boto3.client("s3")
    s3_objs = s3.list_objects(Bucket=Bucket.Datasets.value, MaxKeys=1)
    assert "Contents" in s3_objs
    assert len(s3_objs["Contents"]) == 1
    obj = s3_objs["Contents"][0]
    key = obj["Key"]
    return key


def test_get_s3_data_success_active_user(
    client: TestClient, normal_user_token_headers: dict[str, str], s3_existing_key: str
):

    res = client.get(
        "/api/v1/data",
        params={"objectKey": s3_existing_key},
        headers=normal_user_token_headers,
    )
    assert res.status_code == status.HTTP_200_OK


def test_get_s3_403_unauthed(client: TestClient, s3_existing_key):
    res = client.get("/api/v1/data", params={"objectKey": s3_existing_key})
    assert res.status_code == status.HTTP_401_UNAUTHORIZED
