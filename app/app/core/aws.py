import enum
import io

import boto3
import pandas as pd
from fastapi.datastructures import UploadFile

from app.core.config import settings


class Bucket(enum.Enum):
    Datasets = settings.AWS_DATASETS
    Models = settings.AWS_MODELS


def create_s3_client():
    s3 = boto3.client(
        "s3",
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_SECRET_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_KEY,
    )
    return s3


def upload_s3_file(file: UploadFile, bucket: Bucket, key):
    s3 = create_s3_client()
    s3.upload_fileobj(file.file, bucket.value, key)


def download_file_as_dataframe(bucket: Bucket, key: str) -> pd.DataFrame:
    s3 = create_s3_client()
    s3_res = s3.get_object(Bucket=bucket.value, Key=key)
    s3body = s3_res["Body"].read()
    data = io.BytesIO(s3body)
    df = pd.read_csv(data)
    return df


def delete_s3_file(key: str, bucket: Bucket):
    s3 = create_s3_client()
    s3.delete_object(Bucket=bucket.value, Key=key)
