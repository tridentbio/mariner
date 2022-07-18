import datetime
import io
import json

import pandas
from fastapi.datastructures import UploadFile
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm.session import Session

from app.core.aws import Bucket, upload_s3_file
from app.core.config import settings
from app.features.dataset.exceptions import DatasetNotFound, NotCreatorOfDataset
from app.utils import hash_md5

from ..user.model import User
from .crud import repo
from .schema import (
    ColumnDescription,
    ColumnMetadata,
    ColumnsMeta,
    Dataset,
    DatasetCreate,
    DatasetCreateRepo,
    DatasetsQuery,
    DatasetUpdate,
    DatasetUpdateRepo,
    DataType,
)
from .utils import get_stats

DATASET_BUCKET = settings.AWS_DATASETS


def make_key(filename: str):
    return hash_md5(file=filename)


def get_my_datasets(db: Session, current_user: User, query: DatasetsQuery):
    query.created_by_id = current_user.id
    datasets, total = repo.get_many_paginated(db, query)
    return datasets, total


def get_my_dataset_by_id(db: Session, current_user: User, dataset_id: int):
    dataset = repo.get(db, dataset_id)
    if dataset is None:
        raise DatasetNotFound()
    if current_user.id != dataset.created_by_id:
        raise NotCreatorOfDataset()
    return dataset


def _get_entity_info_from_csv(file: UploadFile):
    file_bytes = file.file.read()
    df = pandas.read_csv(io.BytesIO(file_bytes))
    return len(df), len(df.columns), len(file_bytes), get_stats(df).to_dict()


def _upload_s3(file: UploadFile):
    file_md5 = make_key(file)
    key = f"datasets/{file_md5}.csv"
    upload_s3_file(file, Bucket.Datasets, key)
    return key


def create_dataset(db: Session, current_user: User, data: DatasetCreate):
    rows, columns, bytes, stats = _get_entity_info_from_csv(data.file)
    data.file.file.seek(0)
    data_url = _upload_s3(data.file)
    create_obj = DatasetCreateRepo(
        columns=columns,
        rows=rows,
        split_actual=None,
        split_target=data.split_target,
        split_type=data.split_type,
        name=data.name,
        description=data.description,
        bytes=bytes,
        created_at=datetime.datetime.now(),
        updated_at=datetime.datetime.now(),
        stats=stats if isinstance(stats, dict) else jsonable_encoder(stats),
        data_url=data_url,
        created_by_id=current_user.id,
    )
    if data.columns_descriptions:
        parsed = [json.loads(description) for description in data.columns_descriptions]
        create_obj.columns_descriptions = [
            ColumnDescription(pattern=c["pattern"], description=c["description"])
            for c in parsed
        ]
    if data.columns_metadata:
        parsed = [json.loads(metadata) for metadata in data.columns_metadata]
        create_obj.columns_metadatas = [
            ColumnMetadata(key=m["key"], data_type=DataType(m["data_type"]))
            for m in parsed
        ]
    dataset = repo.create(db, create_obj)
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


def parse_csv_headers(csv_file: UploadFile):
    _, _, _, stats = _get_entity_info_from_csv(csv_file)
    metadata = [
        ColumnsMeta(name=key, nacount=stats[key]["na_count"], dtype=stats[key]["types"])
        for key in stats
    ]
    return metadata
