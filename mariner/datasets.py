import datetime
import io
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
from mariner.schemas.dataset_schemas import (
    CategoricalDataType,
    ColumnsMeta,
    Dataset,
    DatasetCreate,
    DatasetCreateRepo,
    DatasetsQuery,
    DatasetUpdate,
    DatasetUpdateRepo,
    NumericalDataType,
    SmileDataType,
    StringDataType,
)
from mariner.stats import get_metadata as get_stats
from mariner.stats import get_stats as get_summary
from mariner.stores.dataset_sql import dataset_store
from mariner.utils import hash_md5
from mariner.validation import is_valid_smiles_series
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


def _get_df_with_split_indexs(
    df: pd.DataFrame,
    split_type: Literal["random", "scaffold"],
    split_target: str,
    split_column: Union[str, None] = None,
):
    train_size, val_size, test_size = map(lambda x: int(x)/100, split_target.split("-"))
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


def create_dataset(db: Session, current_user: User, data: DatasetCreate):
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

    # All dataset is loaded into memory
    file_bytes = data.file.file.read()
    filesize = data.file.file.tell()
    df = pd.read_csv(io.BytesIO(file_bytes))
    rows, columns, stats = get_entity_info_from_csv(df)

    df_with_split_indexs = _get_df_with_split_indexs(
        df,
        split_type=data.split_type,
        split_target=data.split_target,
        split_column=data.split_column,
    )
    dataset_file = io.BytesIO()
    df_with_split_indexs.to_csv(dataset_file, index=False)
    data.file.file = dataset_file

    data_url = _upload_s3(data.file)

    # Detect the smiles column name
    smiles_columns = []
    for col in df.columns:
        if is_valid_smiles_series(df[col], weak_check=True):
            smiles_columns.append(col)

    stats = get_summary(df, smiles_columns)

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

    create_obj.stats = stats

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


def infer_domain_type_from_series(series: pd.Series):
    if series.dtype == float:
        return NumericalDataType(domain_kind="numeric")
    elif series.dtype == object:
        # check if it is smiles
        if is_valid_smiles_series(series):
            return SmileDataType(domain_kind="smiles")
        # check if it is likely to be categorical
        series = series.sort_values()
        uniques = series.unique()
        if len(uniques) <= 100:
            return CategoricalDataType(
                domain_kind="categorical",
                classes={val: idx for idx, val in enumerate(uniques)},
            )
        return StringDataType(domain_kind="string")
    elif series.dtype == int:
        series = series.sort_values()
        uniques = series.unique()
        if len(uniques) <= 100:
            return CategoricalDataType(
                domain_kind="categorical",
                classes={val: idx for idx, val in enumerate(uniques)},
            )


def parse_csv_headers(csv_file: UploadFile) -> List[ColumnsMeta]:
    file_bytes = csv_file.file.read()
    df = pd.read_csv(io.BytesIO(file_bytes))
    metadata = [
        ColumnsMeta(name=key, dtype=infer_domain_type_from_series(df[key]))
        for key in df
    ]
    return metadata
