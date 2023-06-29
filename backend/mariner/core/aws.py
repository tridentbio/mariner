"""
AWS service
"""
import enum
import io
from datetime import datetime
from typing import BinaryIO, Tuple, Union

import boto3
import pandas as pd
from botocore.client import BaseClient
from fastapi.datastructures import UploadFile

from mariner.core.config import get_app_settings
from mariner.utils import (
    compress_file,
    get_size,
    hash_md5,
    is_compressed,
    read_compressed_csv,
)


class Bucket(enum.Enum):
    """S3 buckets available to the application"""

    Datasets = get_app_settings().AWS_DATASETS
    Models = get_app_settings().AWS_MODELS


class AWS_Credentials:
    def __init__(
        self,
        expiration: Union[None, datetime],
        access_key_id,
        secret_access_key,
        session_token,
    ):
        self.expiration = expiration
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.session_token = session_token

    def is_expired(self):
        return self.expiration is None or self.expiration < datetime.now()

    def credentials_dict(self):
        return {
            "aws_access_key_id": self.access_key_id,
            "aws_secret_access_key": self.secret_access_key,
            "aws_session_token": self.session_token,
        }


_credentials: Union[AWS_Credentials, None] = None


def _get_new_credentials():
    settings = get_app_settings()
    if settings.AWS_MODE == "local":
        return AWS_Credentials(
            expiration=None,
            access_key_id=settings.AWS_ACCESS_KEY_ID,
            secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            session_token=None,
        )
    else:
        raise ValueError(f"Invalid AWS_MODE: {settings.AWS_MODE}")


def _get_credentials():
    global _credentials  # pylint: disable=W0603
    if _credentials is None or _credentials.is_expired():
        _credentials = _get_new_credentials()
    return _credentials


def create_s3_client() -> BaseClient:
    """Create a boto3 s3 client

    Returns:
        BaseClient: boto3 s3 client
    """
    if get_app_settings().AWS_MODE == "local":
        creds = _get_credentials()
        s3 = boto3.client(
            "s3",
            region_name=get_app_settings().AWS_REGION,
            **creds.credentials_dict(),
        )
    else:
        s3 = boto3.client(
            "s3",
            region_name=get_app_settings().AWS_REGION,
        )
    return s3


def list_s3_objects(bucket: Bucket, prefix: str):
    """List objects from s3 bucket starting with prefix.

    Args:
        bucket: bucket string.
        prefix: The prefix used to search the objects.
    """
    client = create_s3_client()
    response = client.list_objects_v2(
        Bucket=bucket.value,
        Prefix=prefix,
    )
    return response


def upload_s3_file(file: Union[UploadFile, io.BytesIO, BinaryIO], bucket: Bucket, key):
    """Upload a file to S3

    Args:
        file (Union[UploadFile, io.BytesIO]): file to upload
        bucket (Bucket): s3 bucket
        key (_type_): s3 key
    """
    s3 = create_s3_client()
    if not isinstance(file, UploadFile):
        s3.upload_fileobj(file, bucket.value, key)
    else:
        s3.upload_fileobj(file.file, bucket.value, key)


def download_file_as_dataframe(bucket: Union[Bucket, str], key: str) -> pd.DataFrame:
    """Downloads s3 file and attempts to parse it as pd.Dataframe

    Will raise exceptions if object is not in csv format.

    Args:
        bucket: Bucket enum value where the object is located.
        key: The key identifying the object.

    Returns:
        The pandas Dataframe stored in s3.
    """
    s3 = create_s3_client()
    s3_res = s3.get_object(
        Bucket=bucket.value if isinstance(bucket, Bucket) else bucket, Key=key
    )
    s3body = s3_res["Body"].read()
    data = io.BytesIO(s3body)
    df = pd.read_csv(data) if not is_compressed(data) else read_compressed_csv(s3body)
    return df


def delete_s3_file(key: str, bucket: Bucket):
    """Attempts to delete a file from s3.

    Will only work if AWS credentials in project environment have ``s3:DeleteObject`` permissions.

    Args:
        key: key to be deleted.
        bucket: bucket in which object is stored.
    """
    s3 = create_s3_client()
    s3.delete_object(Bucket=bucket.value, Key=key)


def download_s3(key: str, bucket: str) -> io.BytesIO:
    """Download a file from S3

    Args:
        key (str): s3 file key
        bucket (str): s3 bucket

    Returns:
        io.BytesIO: downloaded file
    """
    s3 = create_s3_client()
    s3_res = s3.get_object(Bucket=bucket, Key=key)
    s3body = s3_res["Body"].read()
    return io.BytesIO(s3body)


def upload_s3_compressed(
    file: Union[UploadFile, io.BytesIO, BinaryIO], bucket: Bucket = Bucket.Datasets
) -> Tuple[str, int]:
    """Upload a file to S3 and compress it if it's not already compressed

    Args:
        file (Union[UploadFile, io.BytesIO]): file to upload
        bucket (str, optional): s3 bucket. Defaults to Bucket.Datasets.

    Returns:
        Tuple[str, int]: s3 key and file size in bytes
    """
    file_instance = file if isinstance(file, (io.BytesIO, BinaryIO)) else file.file

    file_size = get_size(file_instance)

    if not is_compressed(file_instance):
        file_instance = compress_file(file_instance)

    file_md5 = hash_md5(data=file_instance.read())
    file_instance.seek(0)
    key = f"datasets/{file_md5}.csv"
    upload_s3_file(file=file_instance, bucket=bucket, key=key)
    return key, file_size
