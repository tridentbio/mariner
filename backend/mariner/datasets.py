"""
Dataset service
"""
import asyncio
import datetime
import json
import logging
from typing import List, Literal, Tuple

from fastapi.datastructures import UploadFile
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm.session import Session
from starlette.responses import ContentStream

from api.websocket import WebSocketMessage, get_websockets_manager
from mariner.core.aws import create_s3_client, download_s3, upload_s3_compressed
from mariner.core.config import settings
from mariner.entities.dataset import Dataset as DatasetEntity
from mariner.entities.user import User
from mariner.exceptions import (
    DatasetAlreadyExists,
    DatasetNotFound,
    NotCreatorOwner,
)
from mariner.ray_actors.dataset_transforms import DatasetTransforms
from mariner.schemas.dataset_schemas import (
    ColumnsDescription,
    ColumnsMeta,
    Dataset,
    DatasetCreate,
    DatasetCreateRepo,
    DatasetProcessStatusEventPayload,
    DatasetsQuery,
    DatasetUpdate,
    DatasetUpdateRepo,
)
from mariner.stores.dataset_sql import dataset_store
from mariner.tasks import TaskView, get_manager
from mariner.utils import is_compressed

DATASET_BUCKET = settings.AWS_DATASETS
LOG = logging.getLogger(__name__)


def get_my_datasets(
    db: Session, current_user: User, query: DatasetsQuery
) -> Tuple[List[Dataset], int]:
    """Fetches datasets owned by the current user

    Args:
        db (Session):
            database session
        current_user (User):
            current user (from token payload)
        query (DatasetsQuery):
            query parameters

    Returns:
        Tuple[List[Dataset], int]: datasets and total number of datasets
    """
    query.created_by_id = current_user.id
    datasets, total = dataset_store.get_many_paginated(db, query)
    return datasets, total


def get_my_dataset_by_id(db: Session, current_user: User, dataset_id: int) -> Dataset:
    """Fetches a dataset owned by the current user by id

    Args:
        db (Session): database session
        current_user (User): current user (from token payload)
        dataset_id (int): dataset id

    Raises:
        DatasetNotFound: if dataset does not exist
        NotCreatorOwner: if current user is not the creator of the dataset

    Returns:
        Dataset: dataset object
    """
    dataset = dataset_store.get(db, dataset_id)
    if dataset is None:
        raise DatasetNotFound()
    if current_user.id != dataset.created_by_id:
        raise NotCreatorOwner()
    return Dataset.from_orm(dataset)


async def process_dataset(
    db: Session, dataset_id: int, columns_metadata: List[ColumnsDescription]
) -> DatasetProcessStatusEventPayload:
    """Processes a dataset by id and columns metadata

       Process occurs in the ray actor and the result is saved in the database

    Args:
        db (Session): database session
        dataset_id (int): dataset id
        columns_metadata (List[ColumnsDescription]):
            list of columns with metadata defined by the user

    Returns:
        DatasetProcessStatusEventPayload:
            object with the dataset and a message describing the result of the process
    """
    dataset = dataset_store.get(db, dataset_id)
    assert dataset
    try:
        file = download_s3(key=dataset.data_url, bucket=DATASET_BUCKET)

        # Send the file to the ray actor by chunks
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

        # Apply split indexes in dataset
        await dataset_ray_transformer.apply_split_indexes.remote(
            split_type=dataset.split_type,
            split_target=dataset.split_target,
            split_column=dataset.split_column,
        )

        # Upload dataset to s3 again with the new split indexes
        data_url, filesize = await dataset_ray_transformer.upload_s3.remote(
            old_data_url=dataset.data_url
        )
        # Check data types of columns and check if columns_metadata is valid
        # If there is no errors, columns of type "categorical" are updated
        (
            columns_metadata,
            errors,
        ) = await dataset_ray_transformer.check_data_types.remote(columns_metadata)

        if errors:
            # If there are errors, the dataset is updated with the errors
            # The errors are sent to the frontend by websocket
            dataset_update = DatasetUpdateRepo(
                id=dataset.id,
                bytes=filesize,
                columns=columns,
                rows=rows,
                data_url=data_url,
                columns_metadata=columns_metadata,
                stats=[],
                updated_at=datetime.datetime.now(),
                ready_status="failed",
                errors=errors,
            )
            dataset = dataset_store.update(db, dataset, dataset_update)
            db.flush()

            error_str = "; ".join(errors["columns"])
            event = DatasetProcessStatusEventPayload(
                dataset_id=dataset.id,
                message=(
                    f"error on dataset creation while checking"
                    f' column types of dataset "{dataset.name}": {error_str}'
                ),
                dataset=dataset,
            )

            return event

        stats = await dataset_ray_transformer.get_dataset_summary.remote(
            columns_metadata
        )

        # If there are no errors, the dataset is updated with the new columns_metadata
        # The dataset is ready to be used and a success message is sent to the frontend
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
            errors=None,
        )

        dataset = dataset_store.update(db, dataset, dataset_update)
        db.flush()
        event = DatasetProcessStatusEventPayload(
            dataset_id=dataset.id,
            message=f'dataset "{dataset.name}" created successfully',
            dataset=dataset,
        )
        return event

    except Exception as e:
        LOG.error(f'Unexpected error while processing dataset "{dataset.name}":\n{e}')
        # Handle unexpected errors
        dataset_update = DatasetUpdateRepo(
            id=dataset.id,
            ready_status="failed",
            errors={"log": ["Unexpected error.", str(e)]},
        )

        dataset = dataset_store.update(db, dataset, dataset_update)
        db.flush()

        event = DatasetProcessStatusEventPayload(
            dataset_id=dataset.id,
            message="Unexpected error while processing dataset.",
            dataset=dataset,
        )
        return event


def start_process(
    db: Session, dataset: DatasetEntity, columns_metadata: List[ColumnsDescription]
):
    """Triggers the processing of a dataset, adding it to the task manager

    When the task is finished, a message is sent to the user via websocket
    All the processing is done in a separate thread
    so the user can continue using the application

    Args:
        db (Session): database session
        dataset (Dataset): dataset to be processed
        columns_metadata (List[ColumnsDescription]):
            list of columns with metadata defined by the user
    """
    task = asyncio.create_task(
        process_dataset(db=db, dataset_id=dataset.id, columns_metadata=columns_metadata)
    )

    def finish_task(task: asyncio.Task, _):
        """Callback function to be called when the task is finished

        Args:
            task (asyncio.Task): task that was finished
            _ (_type_): unused parameter
        """
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


async def create_dataset(
    db: Session, current_user: User, data: DatasetCreate
) -> Dataset:
    """Creates a new dataset and triggers the processing of it

    Args:
        db (Session): database session
        current_user (User): user that is creating the dataset
        data (DatasetCreate): data to create the dataset

    Raises:
        DatasetAlreadyExists: if a dataset with the same name already exists

    Returns:
        Dataset: dataset created with ready_status = "processing"
    """
    existing_dataset = dataset_store.get_by_name(db, data.name)
    if existing_dataset:
        raise DatasetAlreadyExists()

    data_url, filesize = upload_s3_compressed(data.file)

    create_obj = DatasetCreateRepo.construct(
        bytes=filesize,
        data_url=data_url,
        split_actual=None,
        split_target=data.split_target,
        split_type=data.split_type,
        split_column=data.split_column,
        columns_metadata=data.columns_metadata,
        name=data.name,
        description=data.description,
        created_at=datetime.datetime.now(),
        updated_at=datetime.datetime.now(),
        created_by_id=current_user.id,
    )
    dataset = dataset_store.create(db, create_obj)

    start_process(db, dataset, data.columns_metadata)

    return dataset


async def update_dataset(
    db: Session, current_user: User, dataset_id: int, data: DatasetUpdate
) -> Dataset:
    """Updates a dataset with new data and triggers the processing of it

    Args:
        db (Session): database session
        current_user (User): user that is updating the dataset
        dataset_id (int): id of the dataset to be updated
        data (DatasetUpdate): data to update the dataset

    Raises:
        DatasetNotFound: if the dataset does not exist
        NotCreatorOwner: if the user is not the creator of the dataset

    Returns:
        Dataset: dataset updated with ready_status = "processing"
    """
    existingdataset = dataset_store.get(db, dataset_id)

    if not existingdataset:
        raise DatasetNotFound(f"Dataset with id {dataset_id} not found")

    if existingdataset.created_by_id != current_user.id:
        raise NotCreatorOwner("Should be creator of dataset")

    existingdataset.stats = jsonable_encoder(existingdataset.stats)
    dataset_dict = jsonable_encoder(existingdataset)
    update = DatasetUpdateRepo(**dataset_dict)
    needs_processing = False

    if data.name:
        update.name = data.name

    if data.description:
        update.description = data.description

    if data.split_target:
        update.split_target = data.split_target

    if data.split_type:
        update.split_type = data.split_type

    if data.columns_metadata:
        # If the columns metadata is different, we need to process the dataset again
        update.columns_metadata = data.columns_metadata
        update.ready_status = "processing"
        needs_processing = True

    if data.split_column:
        update.split_column = data.split_column

    update.id = dataset_id
    saved = dataset_store.update(db, existingdataset, update)
    db.flush()

    if needs_processing:
        start_process(db, saved, data.columns_metadata)
    return Dataset.from_orm(saved)


def delete_dataset(db: Session, current_user: User, dataset_id: int) -> Dataset:
    """Deletes a dataset from the database

    Args:
        db (Session): database session
        current_user (User): user that is deleting the dataset
        dataset_id (int): id of the dataset to be deleted

    Raises:
        DatasetNotFound: if the dataset does not exist
        NotCreatorOwner: if the user is not the creator of the dataset

    Returns:
        Dataset: dataset deleted
    """

    # TODO - delete from s3
    dataset = dataset_store.get(db, dataset_id)
    if not dataset:
        raise DatasetNotFound(f"Dataset with {dataset_id} not found")
    if dataset.created_by_id != current_user.id:
        raise NotCreatorOwner("Should be creator of dataset")
    dataset = dataset_store.remove(db, dataset.id)
    json.dumps(dataset.stats)
    return Dataset.from_orm(dataset)


async def parse_csv_headers(csv_file: UploadFile) -> List[ColumnsMeta]:
    """Parses the headers of a csv file and returns the best metadata found for each column

    All the parsing is done in the dataset actor

    Args:
        csv_file (UploadFile): csv file to be parsed

    Returns:
        List[ColumnsMeta]: list of metadata for each column in the csv file
    """
    dataset_actor = DatasetTransforms.remote()
    chunk_size = settings.APPLICATION_CHUNK_SIZE
    for chunk in iter(lambda: csv_file.file.read(chunk_size), b""):
        await dataset_actor.write_dataset_buffer.remote(chunk)
    await dataset_actor.set_is_dataset_fully_loaded.remote(True)
    metadata = await dataset_actor.get_columns_metadata.remote()
    return metadata


def get_csv_file(
    db: Session,
    dataset_id: int,
    current_user: User,
    file_type: Literal["original", "error"],
) -> ContentStream:
    """Returns the content stream of requested file of dataset

    Args:
        db (Session): database session
        dataset_id (int): id of the dataset
        current_user (User): user that is requesting the file
        file_type (Literal['original', 'error']): type of the file

    Returns:
        ContentStream: file content
    """
    dataset = dataset_store.get(db, dataset_id)
    if not dataset:
        raise DatasetNotFound(f"Dataset with id {dataset_id} not found")

    if not dataset.created_by_id == current_user.id:
        raise NotCreatorOwner("Should be creator of dataset")

    file_key = None

    if file_type == "original":
        file_key = dataset.data_url

    elif file_type == "error":
        if not dataset.errors:
            raise DatasetNotFound(f"Dataset with id {dataset_id} has no errors")

        file_key = dataset.errors["dataset_error_key"]

    if not file_key:
        raise NotImplementedError(f"File type {file_type} not implemented")

    client = create_s3_client()
    s3_res = client.get_object(Bucket=settings.AWS_DATASETS, Key=file_key)
    return s3_res["Body"].iter_chunks()
