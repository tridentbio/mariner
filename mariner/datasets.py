import asyncio
import datetime
import json
from typing import List

from fastapi.datastructures import UploadFile
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm.session import Session

from api.websocket import WebSocketMessage, get_websockets_manager
from mariner.core.aws import download_s3, upload_s3_compressed
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
    DatasetValidationErrorEventPayload,
)
from mariner.stores.dataset_sql import dataset_store
from mariner.tasks import TaskView, get_manager
from mariner.utils import is_compressed

DATASET_BUCKET = settings.AWS_DATASETS


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


async def process_dataset(
    db: Session,
    dataset_id: int,
    data: DatasetCreate,
) -> DatasetValidationErrorEventPayload:
    dataset = dataset_store.get(db, dataset_id)
    file = download_s3(key=dataset.data_url, bucket=DATASET_BUCKET)

    chunk_size = settings.APPLICATION_CHUNK_SIZE
    dataset_ray_transformer = DatasetTransforms.remote(is_compressed(file))
    for chunk in iter(lambda: file.read(chunk_size), b""):
        await dataset_ray_transformer.write_dataset_buffer.remote(chunk)
    await dataset_ray_transformer.set_is_dataset_fully_loaded.remote(True)

    (
        rows,
        columns,
        _,
    ) = await dataset_ray_transformer.get_entity_info_from_csv.remote()

    await dataset_ray_transformer.apply_split_indexes.remote(
        split_type=data.split_type,
        split_target=data.split_target,
        split_column=data.split_column,
    )

    data_url, filesize = await dataset_ray_transformer.upload_s3.remote(
        old_data_url=dataset.data_url
    )
    stats = await dataset_ray_transformer.get_dataset_summary.remote()
    columns_metadata, errors = await dataset_ray_transformer.check_data_types.remote(
        data.columns_metadata
    )

    if errors:
        dataset_update = DatasetUpdateRepo(
            id=dataset.id,
            bytes=filesize,
            columns=columns,
            rows=rows,
            data_url=data_url,
            columns_metadata=columns_metadata,
            stats=stats if isinstance(stats, dict) else jsonable_encoder(stats),
            updated_at=datetime.datetime.now(),
            ready_status="failed",
        )
        dataset = dataset_store.update(db, dataset, dataset_update)

        error_str = "; ".join(errors["columns"])
        event = DatasetValidationErrorEventPayload(
            dataset_id=dataset.id,
            message=(
                f"error on dataset creation while checking"
                f' column types of dataset "{dataset.name}": {error_str}'
            ),
            error_details=errors,
        )

        return event

    dataset_update = DatasetUpdateRepo(
        id=dataset.id,
        bytes=filesize,
        columns=columns,
        rows=rows,
        data_url=data_url,
        columns_metadata=columns_metadata,
        stats=stats if isinstance(stats, dict) else jsonable_encoder(stats),
        updated_at=datetime.datetime.now(),
        ready_status="ready",
    )

    dataset = dataset_store.update(db, dataset, dataset_update)
    event = DatasetValidationErrorEventPayload(
        dataset_id=dataset.id, message=f'dataset "{dataset.name}" created successfully'
    )
    return event


async def create_dataset(db: Session, current_user: User, data: DatasetCreate):
    existing_dataset = dataset_store.get_by_name(db, data.name)
    if existing_dataset:
        raise DatasetAlreadyExists()

    data_url, filesize = upload_s3_compressed(data.file)

    create_obj = DatasetCreateRepo(
        bytes=filesize,
        data_url=data_url,
        split_actual=None,
        split_target=data.split_target,
        split_type=data.split_type,
        name=data.name,
        description=data.description,
        created_at=datetime.datetime.now(),
        updated_at=datetime.datetime.now(),
        created_by_id=current_user.id,
    )
    dataset = dataset_store.create(db, create_obj)
    create_obj.id = dataset.id

    task = asyncio.create_task(
        process_dataset(
            db=db,
            dataset_id=dataset.id,
            data=data,
        )
    )

    def finish_task(task: asyncio.Task, _):
        asyncio.ensure_future(
            get_websockets_manager().send_message(
                user_id=dataset.created_by_id,
                message=WebSocketMessage(
                    data=task.result(), type="dataset-process-finish"
                ),
            )
        )

    get_manager("dataset").add_new_task(
        TaskView(id=dataset.id, user_id=dataset.created_by_id, task=task),
        finish_task,
    )

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
    dataset_actor = DatasetTransforms.remote()
    chunk_size = settings.APPLICATION_CHUNK_SIZE
    for chunk in iter(lambda: csv_file.file.read(chunk_size), b""):
        await dataset_actor.write_dataset_buffer.remote(chunk)
    await dataset_actor.set_is_dataset_fully_loaded.remote(True)
    metadata = await dataset_actor.get_columns_metadata.remote()
    return metadata
