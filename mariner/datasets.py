import datetime
import json
from typing import Any, List, Literal, Mapping, Union

import pandas as pd
from fastapi.datastructures import UploadFile
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm.session import Session

from mariner.core.aws import Bucket, upload_s3_file
from mariner.core.config import settings
from mariner.entities.user import User
from mariner.exceptions import (
    DatasetAlreadyExists,
    DatasetNotFound,
    NotCreatorOwner,
)
from mariner.ray_actors.dataset_transforms import DatasetTransforms
from mariner.schemas.dataset_schemas import (
    ColumnsMeta,
    Dataset,
    DatasetCreate,
    DatasetCreateRepo,
    DatasetsQuery,
    DatasetUpdate,
    DatasetUpdateRepo,
)
from mariner.stats import get_metadata as get_stats
from mariner.stores.dataset_sql import dataset_store
from mariner.utils import hash_md5
from model_builder.splitters import RandomSplitter, ScaffoldSplitter

DATASET_BUCKET = settings.AWS_DATASETS


def make_key(filename: Union[str, UploadFile]):
    return hash_md5(file=filename)


def get_my_datasets(db: Session, current_user: User, query: DatasetsQuery):
    query.created_by_id = current_user.id
    datasets, total = dataset_store.get_many_paginated(db, query)
    return datasets, total


def get_my_dataset_by_id(db: Session, current_user: User, dataset_id: int):
    dataset = dataset_store.get(db, dataset_id)
    if dataset is None:
        raise DatasetNotFound()
    if current_user.id != dataset.created_by_id:
        raise NotCreatorOwner()
    return dataset


def get_entity_info_from_csv(
    df: pd.DataFrame,
) -> tuple[int, int, Mapping[Any, Any]]:
    stats: Mapping[Any, Any] = get_stats(df).to_dict(orient="dict")
    return len(df), len(df.columns), stats


def _upload_s3(file: UploadFile):
    file.file.seek(0)
    file_md5 = make_key(file)
    key = f"datasets/{file_md5}.csv"
    file.file.seek(0)
    upload_s3_file(file, Bucket.Datasets, key)
    return key


def _get_df_with_split_indexes(
    df: pd.DataFrame,
    split_type: Literal["random", "scaffold"],
    split_target: str,
    split_column: Union[str, None] = None,
):
    train_size, val_size, test_size = map(
        lambda x: int(x) / 100, split_target.split("-")
    )
    if split_type == "random":
        splitter = RandomSplitter()
        df = splitter.split(df, train_size, test_size, val_size)
        return df
    elif split_type == "scaffold":
        splitter = ScaffoldSplitter()
        assert (
            split_column is not None
        ), "split column can't be none when split_type is scaffold"
        df = splitter.split(df, split_column, train_size, test_size, val_size)
        return df
    else:
        raise NotImplementedError(f"{split_type} splitting is not implemented")


async def create_dataset(db: Session, current_user: User, data: DatasetCreate):
    """
    Domain function that creates a dataset for the user.

    :param db Session: Connection to the database
    :param current_user User: request owner
    :param data DatasetCreate: dataset attributes
    :raises DatasetAlreadyExists: when there is a name clash between the user datasets
    :raises ValueError: When data.split_column is not provided and data.split_type is
    scaffold
    """
    existing_dataset = dataset_store.get_by_name(db, data.name)
    if existing_dataset:
        raise DatasetAlreadyExists()
    dataset_ray_transformer = DatasetTransforms.remote(file_input=data.file.file)
    (
        rows,
        columns,
        filesize,
        stats,
    ) = await dataset_ray_transformer.get_entity_info_from_csv.remote()
    await dataset_ray_transformer.apply_split_indexes.remote(
        split_type=data.split_type,
        split_target=data.split_target,
        split_column=data.split_column,
    )
    data_url = await dataset_ray_transformer.upload_s3.remote()
    stats = await dataset_ray_transformer.get_dataset_summary.remote()
    create_obj = DatasetCreateRepo(
        columns=columns,
        rows=rows,
        split_actual=None,
        split_target=data.split_target,
        split_type=data.split_type,
        name=data.name,
        description=data.description,
        bytes=filesize,
        created_at=datetime.datetime.now(),
        updated_at=datetime.datetime.now(),
        stats=stats if isinstance(stats, dict) else jsonable_encoder(stats),
        data_url=data_url,
        created_by_id=current_user.id,
        columns_metadata=data.columns_metadata,
    )
    dataset = dataset_store.create(db, create_obj)
    return dataset


def update_dataset(
    db: Session, current_user: User, dataset_id: int, data: DatasetUpdate
):
    existingdataset = dataset_store.get(db, dataset_id)

    if not existingdataset:
        raise DatasetNotFound(f"Dataset with id {dataset_id} not found")

    if existingdataset.created_by_id != current_user.id:
        raise NotCreatorOwner("Should be creator of dataset")

    existingdataset.stats = jsonable_encoder(existingdataset.stats)
    dataset_dict = jsonable_encoder(existingdataset)
    update = DatasetUpdateRepo(**dataset_dict)

    if data.name:
        update.name = data.name

    if data.description:
        update.description = data.description

    if data.split_target:
        update.split_target = data.split_target

    if data.split_type:
        update.split_type = data.split_type

    if data.columns_metadata:
        update.columns_metadata = data.columns_metadata

    if data.split_column:
        update.split_column = data.split_column

    update.id = dataset_id
    saved = dataset_store.update(db, existingdataset, update)
    db.flush()
    return Dataset.from_orm(saved)


def delete_dataset(db: Session, current_user: User, dataset_id: int):
    dataset = dataset_store.get(db, dataset_id)
    if not dataset:
        raise DatasetNotFound(f"Dataset with {dataset_id} not found")
    if dataset.created_by_id != current_user.id:
        raise NotCreatorOwner("Should be creator of dataset")
    dataset = dataset_store.remove(db, dataset.id)
    json.dumps(dataset.stats)
    return Dataset.from_orm(dataset)


# Candidate for going to dataset actor
async def parse_csv_headers(csv_file: UploadFile) -> List[ColumnsMeta]:
    dataset_actor = DatasetTransforms.remote(file_input=csv_file.file)
    metadata = await dataset_actor.get_columns_metadata.remote()
    return metadata
