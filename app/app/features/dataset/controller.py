import datetime
import io
import json
from typing import Any, List, Mapping, Union

import pandas as pd
from fastapi.datastructures import UploadFile
from fastapi.encoders import jsonable_encoder
from rdkit import Chem
from sqlalchemy.orm.session import Session

from app.builder.splitters import RandomSplitter, ScaffoldSplitter
from app.core.aws import Bucket, upload_s3_file
from app.core.config import settings
from app.features.dataset.exceptions import (
    DatasetAlreadyExists,
    DatasetNotFound,
    NotCreatorOfDataset,
)
from app.utils import hash_md5

from .stats import get_stats as get_summary
from ..user.model import User
from .crud import repo
from .schema import (
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
from .utils import get_stats

DATASET_BUCKET = settings.AWS_DATASETS


def make_key(filename: Union[str, UploadFile]):
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
    file: UploadFile,
) -> tuple[int, int, int, Mapping[Any, Any]]:
    file_bytes = file.file.read()
    df = pd.read_csv(io.BytesIO(file_bytes))
    assert isinstance(df, pd.DataFrame)
    stats = get_stats(df).to_dict()
    return len(df), len(df.columns), len(file_bytes), stats


def _upload_s3(file: UploadFile):
    file_md5 = make_key(file)
    key = f"datasets/{file_md5}.csv"
    file.file.seek(0)
    upload_s3_file(file, Bucket.Datasets, key)
    return key


def create_dataset(db: Session, current_user: User, data: DatasetCreate):
    existing_dataset = repo.get_by_name(db, data.name)

    if existing_dataset:
        raise DatasetAlreadyExists()
    rows, columns, bytes, stats = get_entity_info_from_csv(data.file)
    data.file.file.seek(0)

    # Before upload we need to do the split
    if data.split_type == "random":
        splitter = RandomSplitter()

        file_bytes = data.file.file.read()
        df = pd.read_csv(io.BytesIO(file_bytes))

        split_target = data.split_target.split("-")
        train_size, val_size, test_size = split_target
        train_size = int(train_size) / 100
        val_size = int(val_size) / 100
        test_size = int(test_size) / 100

        df = splitter.split(df, train_size, test_size, val_size)

        dataset_file = io.BytesIO()
        df.to_csv(dataset_file)

        data.file.file = dataset_file
    else:

        if data.split_column is None:
            raise ValueError(
                "Split Column cannot be none due to ScaffoldSplitter"
            )

        splitter = ScaffoldSplitter()

        file_bytes = data.file.file.read()
        df = pd.read_csv(io.BytesIO(file_bytes))

        split_target = data.split_target.split("-")
        train_size, val_size, test_size = split_target
        train_size = int(train_size) / 100
        val_size = int(val_size) / 100
        test_size = int(test_size) / 100

        df = splitter.split(
            df,
            data.split_column,
            train_size,
            test_size,
            val_size
        )

        dataset_file = io.BytesIO()
        df.to_csv(dataset_file)

        data.file.file = dataset_file

    # Detect the smiles column name
    smiles_column = None

    for col in df.columns:
        if validate_smiles_series(df[col]):
            smiles_column = col
            break

    if smiles_column:
        stats = get_summary(df, smiles_column)

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

    if smiles_column:
        create_obj.stats = stats

    dataset = repo.create(db, create_obj)
    dataset = repo.get(db, dataset.id)

    return dataset


def update_dataset(
    db: Session, current_user: User, dataset_id: int, data: DatasetUpdate
):
    existingdataset = repo.get(db, dataset_id)

    if not existingdataset:
        raise DatasetNotFound(f"Dataset with id {dataset_id} not found")

    if existingdataset.created_by_id != current_user.id:
        raise NotCreatorOfDataset("Should be creator of dataset")

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
        if data.split_type == "random":
            data.file.file.seek(0)

            splitter = RandomSplitter()

            file_bytes = data.file.file.read()
            df = pd.read_csv(io.BytesIO(file_bytes))

            split_target = data.split_target.split("-")
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
                    "Split Column cannot be none due to ScaffoldSplitter"
                )

            splitter = ScaffoldSplitter()
            file_bytes = data.file.file.read()
            df = pd.read_csv(io.BytesIO(file_bytes))
            split_target = data.split_target.split("-")
            train_size, val_size, test_size = split_target
            train_size = int(train_size) / 100
            val_size = int(val_size) / 100
            test_size = int(test_size) / 100
            dataset = splitter.split(
                df, data.split_column, train_size, test_size, val_size
            )

            dataset_file = io.BytesIO()
            dataset.to_csv(dataset_file)
            dataset_file.seek(0)
            data.file.file = dataset_file

        update.data_url = _upload_s3(data.file)

        # Detect the smiles column name
        smiles_column = None

        for col in df.columns:
            if validate_smiles_series(df[col]):
                smiles_column = col
                break

        if smiles_column:
            stats = get_summary(dataset, smiles_column)

        if smiles_column:
            update.stats = stats

    update.id = dataset_id
    saved = repo.update(db, existingdataset, update)
    db.flush()
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


def validate_smiles(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    except Exception:
        raise ValueError(f'Type of SMILES {type(smiles)} must be a string.')

    if mol is None:
        raise ValueError(f'SMILES "{smiles}" is not syntacticaly valid.')
    else:
        try:
            Chem.SanitizeMol(mol)
        except:  # noqa: E722
            raise ValueError(
                f'SMILES "{smiles}" does not have valid chemistry.'
            )

    return smiles


def validate_smiles_series(smiles_series: pd.Series) -> bool:
    for val in smiles_series:
        try:
            validate_smiles(val)
        except ValueError:
            return False
    return True


def infer_domain_type_from_series(series: pd.Series):
    if series.dtype == float:
        return NumericalDataType(domain_kind="numerical")
    elif series.dtype == object:
        # check if it is smiles
        if validate_smiles_series(series):
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
