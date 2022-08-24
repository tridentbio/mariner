import datetime
import io
import json
from typing import List

import pandas
from fastapi.datastructures import UploadFile
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm.session import Session

from app.builder.splitters import RandomSplitter, ScaffoldSplitter
from app.core.aws import Bucket, upload_s3_file
from app.core.config import settings
from app.features.dataset.exceptions import (
    DatasetAlreadyExists,
    DatasetNotFound,
    InvalidCategoricalColumn,
    NotCreatorOfDataset,
)
from app.utils import hash_md5

from ..user.model import User
from .crud import repo
from .schema import (
    ColumnsDescription,
    ColumnsMeta,
    Dataset,
    DatasetCreate,
    DatasetCreateRepo,
    DatasetsQuery,
    DatasetUpdate,
    DatasetUpdateRepo,
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


def get_entity_info_from_csv(
    file: UploadFile, columns_descriptions: List[ColumnsDescription]
):
    file_bytes = file.file.read()
    df = pandas.read_csv(io.BytesIO(file_bytes))
    assert isinstance(df, pandas.DataFrame)
    stats = get_stats(df).to_dict()
    import re

    for column in stats:
        for description in columns_descriptions:
            if description.data_type == "categorical" and re.compile(
                description.pattern
            ).search(str(column)):
                if df[column].dtype not in [int, object]:
                    raise InvalidCategoricalColumn(
                        f'Column "{column}" of type{df[column].dtype} cannot'
                        "be a categorical column because it's type is not string"
                        "or int"
                    )
                else:
                    unique_values = df[column].unique()
                    categoriesmap = {
                        column: index for index, column in enumerate(unique_values)
                    }
                    stats[column]["categories"] = categoriesmap
    return len(df), len(df.columns), len(file_bytes), stats


def _upload_s3(file: UploadFile):
    file_md5 = make_key(file)
    key = f"datasets/{file_md5}.csv"
    upload_s3_file(file, Bucket.Datasets, key)
    return key


def create_dataset(db: Session, current_user: User, data: DatasetCreate):
    existing_dataset = repo.get_by_name(db, data.name)

    if existing_dataset:
        raise DatasetAlreadyExists()
    rows, columns, bytes, stats = get_entity_info_from_csv(
        data.file, data.columns_metadata
    )
    data.file.file.seek(0)

    # Before upload we need to do the split
    if data.split_type == 'random':
        splitter = RandomSplitter()

        file_bytes = data.file.file.read()
        df = pandas.read_csv(io.BytesIO(file_bytes))

        split_target = data.split_target.split('-')
        train_size, val_size, test_size = split_target
        train_size = int(train_size) / 100
        val_size = int(val_size) / 100
        test_size = int(test_size) / 100

        dataset = splitter.split(df, train_size, test_size, val_size)

        dataset_file = io.BytesIO()
        dataset.to_csv(dataset_file)
        dataset_file.seek(0)

        data.file.file = dataset_file
    else:

        if data.split_column is None:
            raise ValueError(
                'Split Column cannot be none due to ScaffoldSplitter'
            )

        splitter = ScaffoldSplitter()

        file_bytes = data.file.file.read()
        df = pandas.read_csv(io.BytesIO(file_bytes))

        split_target = data.split_target.split('-')
        train_size, val_size, test_size = split_target
        train_size = int(train_size) / 100
        val_size = int(val_size) / 100
        test_size = int(test_size) / 100

        dataset = splitter.split(
            df,
            data.split_column,
            train_size,
            test_size,
            val_size
        )

        dataset_file = io.BytesIO()
        dataset.to_csv(dataset_file)
        dataset_file.seek(0)

        data.file.file = dataset_file

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
        columns_metadata=data.columns_metadata,
    )

    dataset = repo.create(db, create_obj)
    dataset = repo.get(db, dataset.id)
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

    if data.columns_metadata:
        update.columns_metadata = data.columns_metadata

    if data.split_column:
        update.split_column = data.split_column

    if data.file:
        file_bytes = data.file.file.read()
        (
            update.rows,
            update.columns,
            update.bytes,
            update.stats,
        ) = get_entity_info_from_csv(data.file)
        data.file.file.seek(0)

        # Before upload we need to do the split
        if data.split_type == 'random':
            data.file.file.seek(0)

            splitter = RandomSplitter()

            file_bytes = data.file.file.read()
            df = pandas.read_csv(io.BytesIO(file_bytes))

            split_target = data.split_target.split('-')
            train_size, val_size, test_size = split_target
            train_size = int(train_size) / 100
            val_size = int(val_size) / 100
            test_size = int(test_size) / 100

            dataset = splitter.split(df, train_size, test_size, val_size)

            dataset_file = io.BytesIO()
            dataset.to_csv(dataset_file)
            dataset_file.seek(0)

            data.file.file = dataset_file

        else:

            if data.split_column is None:
                raise ValueError(
                    'Split Column cannot be none due to ScaffoldSplitter'
                )

            splitter = ScaffoldSplitter()

            file_bytes = data.file.file.read()
            df = pandas.read_csv(io.BytesIO(file_bytes))

            split_target = data.split_target.split('-')
            train_size, val_size, test_size = split_target
            train_size = int(train_size) / 100
            val_size = int(val_size) / 100
            test_size = int(test_size) / 100

            dataset = splitter.split(
                df,
                data.split_column,
                train_size,
                test_size,
                val_size
            )

            dataset_file = io.BytesIO()
            dataset.to_csv(dataset_file)
            dataset_file.seek(0)

            data.file.file = dataset_file

        update.data_url = _upload_s3(data.file)

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

    file_bytes = csv_file.file.read()
    df = pandas.read_csv(io.BytesIO(file_bytes))
    metadata = [
        ColumnsMeta(name=key, nacount=0, dtype=str(df[key].dtype)) for key in df
    ]
    return metadata
