import datetime
import io
from uuid import uuid4

import boto3
import pandas
from fastapi.datastructures import UploadFile
from fastapi.exceptions import HTTPException
from sqlalchemy.orm.session import Session

from ..user.schema import User
from .crud import repo
from .schema import (
    DatasetCreate,
    DatasetCreateRepo,
    DatasetsQuery,
    DatasetUpdate,
    DatasetUpdateRepo,
)

# TODO: move to somewhere appropriate
DATASET_BUCKET = "datasets-bucket"


def make_key():
    return str(uuid4())


def get_my_datasets(db: Session, current_user: User, query: DatasetsQuery):
    if not current_user.id:
        raise Exception("wtf")
    query.created_by_id = current_user.id
    datasets, total = repo.get_many_paginated(db, query)
    return datasets, total


def _get_stats(df):
    stats = df.describe(include="all").to_dict()
    return stats


def _get_entity_info_from_csv(file_bytes):
    df = pandas.read_csv(io.BytesIO(file_bytes))
    return len(df), len(df.columns), len(file_bytes), _get_stats(df)


def _upload_s3(file: UploadFile):
    key = make_key()
    s3 = boto3.client("s3")
    # TODO: Here we should encrypt if bucket is not encrypted by AWS already
    s3.upload_fileobj(file, DATASET_BUCKET, key)
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
            stats=stats,
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
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset.created_by_id != current_user.id:
        raise HTTPException(
            status_code=400, detail="Can only update datasets you created"
        )

    update = DatasetUpdateRepo(
        name=data.name,
        description=data.description,
        split_target=data.split_target,
        split_type=data.split_type,
    )
    if data.file:
        update.data_url = _upload_s3(data.file)
        file_bytes = data.file.file.read()
        (
            update.rows,
            update.columns,
            update.bytes,
            update.stats,
        ) = _get_entity_info_from_csv(file_bytes)

    dataset = repo.update(db, dataset, update)
    return dataset


def delete_dataset(db: Session, current_user: User, dataset_id: int):
    dataset = repo.get(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset.created_by_id != current_user.id:
        raise HTTPException(
            status_code=400, detail="Can only update datasets you created"
        )
    dataset = repo.remove(db, dataset.id)
    return dataset
