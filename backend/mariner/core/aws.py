"""
AWS service
"""
import enum
import io
import os
from datetime import datetime
from typing import BinaryIO, Optional, Tuple, Union

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError
from fastapi.datastructures import UploadFile

from fleet.file_utils import is_compressed
from mariner.core.config import get_app_settings
from mariner.utils import compress_file, get_size, hash_md5


class Bucket(enum.Enum):
    """S3 buckets available to the application"""

    Datasets = get_app_settings("secrets").aws_datasets
    Models = get_app_settings("secrets").aws_models


class AWS_Credentials:
    """
    Represents AWS credentials with utility methods.
    """

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
        """
        Checks if it's expired.
        """
        return self.expiration is None or self.expiration < datetime.now()

    def credentials_dict(self):
        """
        Returns a dictionary with boto clients credentials params.
        """
        return {
            "aws_access_key_id": self.access_key_id,
            "aws_secret_access_key": self.secret_access_key,
            "aws_session_token": self.session_token,
        }


_credentials: Union[AWS_Credentials, None] = None


def _get_new_credentials():
    secrets = get_app_settings("secrets")
    if secrets.aws_mode == "local":
        return AWS_Credentials(
            expiration=None,
            access_key_id=secrets.aws_access_key_id,
            secret_access_key=secrets.aws_secret_access_key,
            session_token=None,
        )
    else:
        raise ValueError(f"Invalid aws_mode: {secrets.aws_mode}")


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
    if get_app_settings().secrets.aws_mode == "local":
        creds = _get_credentials()
        s3 = boto3.client(
            "s3",
            region_name=get_app_settings().secrets.aws_region,
            **creds.credentials_dict(),
        )
    else:
        s3 = boto3.client(
            "s3",
            region_name=get_app_settings().secrets.aws_region,
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


def upload_s3_file(
    file: Union[UploadFile, io.BytesIO, BinaryIO], bucket: Bucket, key
):
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


def delete_s3_file(key: str, bucket: Bucket):
    """Attempts to delete a file from s3.

    Will only work if AWS credentials in project environment have ``s3:DeleteObject`` permissions.

    Args:
        key: key to be deleted.
        bucket: bucket in which object is stored.
    """
    s3 = create_s3_client()
    s3.delete_object(Bucket=bucket.value, Key=key)


def download_s3(
    key: str, bucket: Union[str, Bucket], range_: Optional[str] = None
) -> io.BytesIO:
    """Download a file from S3

    Args:
        key (str): s3 file key
        bucket (str): s3 bucket
        range (str): A range as specified by rfc9110

    Returns:
        io.BytesIO: downloaded file


    See Also:
        Reference for range spec: https://www.rfc-editor.org/rfc/rfc9110.html#name-range
    """
    s3 = create_s3_client()
    kwargs = {
        "Bucket": bucket.value if isinstance(bucket, Bucket) else bucket,
        "Key": key,
    }
    if range_:
        kwargs["Range"] = range_
    s3_res = s3.get_object(**kwargs)
    s3body = s3_res["Body"].read()
    return io.BytesIO(s3body)


def upload_s3_compressed(
    file: Union[UploadFile, io.BytesIO, BinaryIO],
    bucket: Bucket = Bucket.Datasets,
) -> Tuple[str, int]:
    """Upload a file to S3 and compress it if it's not already compressed

    Args:
        file (Union[UploadFile, io.BytesIO]): file to upload
        bucket (str, optional): s3 bucket. Defaults to Bucket.Datasets.

    Returns:
        Tuple[str, int]: s3 key and file size in bytes
    """
    file_instance = (
        file if isinstance(file, (io.BytesIO, BinaryIO)) else file.file
    )

    file_size = get_size(file_instance)

    if not is_compressed(file_instance):
        file_instance = compress_file(file_instance)

    file_md5 = hash_md5(data=file_instance.read())
    file_instance.seek(0)
    key = f"datasets/{file_md5}.csv"
    upload_s3_file(file=file_instance, bucket=bucket, key=key)
    return key, file_size


def is_in_s3(key: str, bucket: Bucket) -> bool:
    """Check if a file is in S3

    Args:
        key (str): s3 file key
        bucket (Bucket): s3 bucket

    Returns:
        bool: True if file is in S3, False otherwise
    """
    s3 = create_s3_client()
    try:
        s3.head_object(Bucket=bucket.value, Key=key)
        return True
    except (s3.exceptions.NoSuchKey, ClientError):
        return False


def download_head(
    bucket: str, key: str, nlines: int = 3, chunk_size=int(1.6e4)
):
    """
    Downloads at least the first nlines of the S3 object described by bucket and key.

    If more lines than nlines are returned, you shouldn't trust they are complete.
    This function is not recommended to install the full dataset (for that use download_s3).

    Args:
        bucket (str): The bucket in which the object is stored.
        key (str): The object identifier.
        nlines (int): The number of full lines to download. Defaults to 3.
        chunk_size (int): The number of chunks to read at a time. Defaults to 1.6e4 (16 kilobytes).
    """

    start, end = 0, chunk_size - 1
    data = io.StringIO()
    len_data = 0
    while (
        len_data < nlines + 1
    ):  # Get one extra since it may be incomplete (discard it later)
        res = download_s3(
            bucket=bucket, key=key, range_=f"bytes={start}-{end}"
        )
        # We decode the bytes chunk into utf-8 for safe line counting
        # Make it a bit faster by not decoding intoi utf-8 and counting
        # line feed bytes https://en.wikipedia.org/wiki/Newline#Unicode
        decoded = res.read().decode(encoding="utf-8")
        len_data += decoded.count("\n")
        data.write(decoded)
        start += chunk_size
        end += chunk_size
    data.seek(0, os.SEEK_SET)
    return data
