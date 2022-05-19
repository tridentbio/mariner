import datetime
import io
from uuid import uuid4

import pandas
from fastapi.datastructures import UploadFile
from fastapi.encoders import jsonable_encoder
from pandas.core.frame import DataFrame
from sqlalchemy.orm.session import Session

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


def _get_stats(df) -> DataFrame:
    stats = df.describe(include="all")
    return stats


def _get_entity_info_from_csv(file_bytes):
    df = pandas.read_csv(io.BytesIO(file_bytes))
    return len(df), len(df.columns), len(file_bytes), str(_get_stats(df).to_json())


def _upload_s3(file: UploadFile):
    key = make_key()
    # s3 = boto3.client("s3")
    # TODO: Here we should encrypt if bucket is not encrypted by AWS already
    # s3.upload_fileobj(file, DATASET_BUCKET, key)
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
        raise DatasetNotFound(f"Dataset with id {dataset_id} not found")
    if dataset.created_by_id != current_user.id:
        raise NotCreatorOfDataset("Should be creator of dataset")
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
    return dataset
