import datetime
import io
import json
from uuid import uuid4

import boto3
import pandas
from fastapi.datastructures import UploadFile
from fastapi.encoders import jsonable_encoder
from pandas.core.frame import DataFrame
from sqlalchemy.orm.session import Session

from app.core.config import settings
from app.features.dataset.exceptions import DatasetNotFound, NotCreatorOfDataset

from ..user.model import User
from .crud import repo
from .schema import (
    Dataset,
    DatasetCreate,
    DatasetCreateRepo,
    DatasetsQuery,
    DatasetUpdate,
    DatasetUpdateRepo,
)

DATASET_BUCKET = settings.AWS_DATASETS_BUCKET


def make_key():
    return str(uuid4())


def get_my_datasets(db: Session, current_user: User, query: DatasetsQuery):
    query.created_by_id = current_user.id
    datasets, total = repo.get_many_paginated(db, query)
    return datasets, total


def get_dataset_by_id(db: Session, current_user: User, dataset_id: int):
    dataset = repo.get(db, dataset_id)
    if dataset is None:
        raise DatasetNotFound()
    if current_user.id != dataset.created_by_id:
        raise NotCreatorOfDataset()
    return dataset


def _get_stats(df) -> DataFrame:
    stats = df.describe(include="all").fillna("NaN")
    return stats


def _get_entity_info_from_csv(file_bytes):
    df = pandas.read_csv(io.BytesIO(file_bytes))
    return len(df), len(df.columns), len(file_bytes), _get_stats(df).to_dict()


def _upload_s3(file: UploadFile):
    key = make_key()

    # print(settings.AWS_REGION, settings.AWS_SECRET_KEY_ID, settings.AWS_SECRET_KEY)
    s3 = boto3.client(
        "s3",
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_SECRET_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_KEY,
    )
    s3.upload_fileobj(file.file, DATASET_BUCKET, key)
    return key


def create_dataset(db: Session, current_user: User, data: DatasetCreate):

    # parse csv bytes as json
    file_raw = data.file.file.read()
    rows, columns, bytes, stats = _get_entity_info_from_csv(file_raw)
    data_url = _upload_s3(data.file)
    dataset = repo.create(
        db,
        DatasetCreateRepo(
            columns=columns,
            rows=rows,
            split_actual=None,
            split_target=data.split_target,
            split_type=data.split_type,
            name=data.name,
            description=data.description,
            bytes=bytes,  # maybe should be the encrypted size instead,
            created_at=datetime.datetime.now(),
            stats=stats if isinstance(stats, dict) else jsonable_encoder(stats),
            data_url=data_url,
            created_by_id=current_user.id,
        ),
    )
    return dataset


def update_dataset(
    db: Session, current_user: User, dataset_id: int, data: DatasetUpdate
):
    dataset = repo.get(db, dataset_id)
    if not dataset:
        raise DatasetNotFound(f"Dataset with id {dataset_id} not found")
    if dataset.created_by_id != current_user.id:
        raise NotCreatorOfDataset("Should be creator of dataset")
    dataset.stats = jsonable_encoder(dataset.stats)
    dataset_dict = jsonable_encoder(dataset)
    update = DatasetUpdateRepo(**dataset_dict)
    if data.name:
        update.name = data.name
    if data.description:
        update.description = data.description
    if data.split_target:
        update.split_target = data.split_target
    if data.split_type:
        update.split_type = data.split_type
    if data.file:
        update.data_url = _upload_s3(data.file)
        file_bytes = data.file.file.read()
        (
            update.rows,
            update.columns,
            update.bytes,
            update.stats,
        ) = _get_entity_info_from_csv(file_bytes)

    update.id = dataset_id
    saved = repo.update(db, dataset, update)
    return Dataset.from_orm(saved)


def delete_dataset(db: Session, current_user: User, dataset_id: int):
    dataset = repo.get(db, dataset_id)
    if not dataset:
        raise DatasetNotFound(f"Dataset with {dataset_id} not found")
    if dataset.created_by_id != current_user.id:
        raise NotCreatorOfDataset("Should be creator of dataset")
    dataset = repo.remove(db, dataset.id)
    json.dumps(dataset.stats)
    return Dataset.from_orm(dataset)
