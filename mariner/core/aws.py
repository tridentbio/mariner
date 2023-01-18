import enum
import io
from typing import Tuple, Union

import boto3
import pandas as pd
from botocore.client import BaseClient
from fastapi.datastructures import UploadFile

from mariner.core.config import settings
from mariner.utils import (
    compress_file,
    get_size,
    hash_md5,
    is_compressed,
    read_compressed_csv,
)


class Bucket(enum.Enum):
    """S3 buckets available to the application"""

    Datasets = settings.AWS_DATASETS
    Models = settings.AWS_MODELS


def create_s3_client() -> BaseClient:
    """Create a boto3 s3 client

    Returns:
        BaseClient: boto3 s3 client
    """
    s3 = boto3.client(
        "s3",
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )
    return s3


def upload_s3_file(file: Union[UploadFile, io.BytesIO], bucket: Bucket, key):
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


def download_file_as_dataframe(bucket: Bucket, key: str) -> pd.DataFrame:
    s3 = create_s3_client()
    s3_res = s3.get_object(Bucket=bucket.value, Key=key)
    s3body = s3_res["Body"].read()
    data = io.BytesIO(s3body)
    df = pd.read_csv(data) if not is_compressed(data) else read_compressed_csv(s3body)
    return df


def delete_s3_file(key: str, bucket: Bucket):
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
    file: Union[UploadFile, io.BytesIO], bucket: Bucket = Bucket.Datasets
) -> Tuple[str, int]:
    """Upload a file to S3 and compress it if it's not already compressed

    Args:
        file (Union[UploadFile, io.BytesIO]): file to upload
        bucket (str, optional): s3 bucket. Defaults to Bucket.Datasets.

    Returns:
        Tuple[str, int]: s3 key and file size in bytes
    """
    file_instance = file if isinstance(file, io.BytesIO) else file.file

    file_size = get_size(file_instance)

    if not is_compressed(file_instance):
        file_instance = compress_file(file_instance)

    file_md5 = hash_md5(data=file_instance.read())
    file_instance.seek(0)
    key = f"datasets/{file_md5}.csv"
    upload_s3_file(file=file_instance, bucket=bucket, key=key)
    return key, file_size
