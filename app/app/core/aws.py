import enum
import io
from typing import Optional

import boto3
import pandas as pd
from fastapi.datastructures import UploadFile

from app.core.config import settings


class Bucket(enum.Enum):
    Datasets = settings.AWS_DATASETS


def create_s3_client():
    s3 = boto3.client(
        "s3",
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_SECRET_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_KEY,
    )
    return s3


def upload_s3_file(file: UploadFile, bucket: Bucket, key: Optional[str] = None):
    s3 = create_s3_client()
    s3.upload_fileobj(file.file, bucket, key)


def download_file_as_dataframe(bucket: Bucket, key: str) -> pd.DataFrame:
    # s3uri = f's3://{bucket.value}/{key}'
    # df = pd.read_csv(s3uri)
    s3 = create_s3_client()
    s3_res = s3.get_object(Bucket=bucket.value, Key=key)
    data = io.BytesIO(s3_res["Body"].read())
    data.seek(0)
    df = pd.read_csv(data)
    return df
