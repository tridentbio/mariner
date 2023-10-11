"""
Handlers for api/v1/datasets* endpoints
"""
from typing import Any, List, Optional

from fastapi.datastructures import UploadFile
from fastapi.exceptions import HTTPException
from fastapi.param_functions import Depends, File, Form
from fastapi.routing import APIRouter
from sqlalchemy.orm.session import Session
from starlette import status
from starlette.responses import StreamingResponse

import mariner.datasets as controller
from api import deps
from mariner.entities.user import User
from mariner.exceptions import (
    DatasetAlreadyExists,
    DatasetNotFound,
    NotCreatorOwner,
)
from mariner.schemas.api import Paginated
from mariner.schemas.dataset_schemas import (
    ColumnMetadataFromJSONStr,
    ColumnsMeta,
    Dataset,
    DatasetCreate,
    DatasetsQuery,
    DatasetSummary,
    DatasetUpdate,
    DatasetUpdateInput,
    Split,
    SplitType,
)
from mariner.utils.metrics import REQUEST_TIME

router = APIRouter()


@REQUEST_TIME.labels(endpoint="/datasets/", method="GET").time()
@router.get("/", response_model=Paginated[Dataset])
def get_my_datasets(
    query: DatasetsQuery = Depends(DatasetsQuery),
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Retrieve datasets owned by requester
    """
    datasets, total = controller.get_my_datasets(db, current_user, query)
    return Paginated(
        data=[Dataset.from_orm(ds) for ds in datasets], total=total
    )


@REQUEST_TIME.labels(
    endpoint="/datasets/{dataset_id}/summary", method="GET"
).time()
@router.get("/{dataset_id}/summary", response_model=DatasetSummary)
def get_my_dataset_summary(
    dataset_id: int,
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Retrieve dataset stats by id
    """
    dataset = controller.get_my_dataset_by_id(db, current_user, dataset_id)
    summary = dataset.stats

    return summary


@REQUEST_TIME.labels(endpoint="/datasets/{dataset_id}", method="GET").time()
@router.get("/{dataset_id}", response_model=Dataset)
def get_my_dataset(
    dataset_id: int,
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Retrieve datasets owned by requester
    """
    dataset = controller.get_my_dataset_by_id(db, current_user, dataset_id)
    return Dataset.from_orm(dataset)


@REQUEST_TIME.labels(endpoint="/datasets/", method="POST").time()
@router.post("/", response_model=Dataset)
async def create_dataset(
    current_user: User = Depends(deps.get_current_active_user),
    name: str = Form(...),
    description: str = Form(...),
    split_target: Split = Form(..., alias="splitTarget"),
    split_type: SplitType = Form(..., alias="splitType"),
    split_column: Optional[str] = Form(None, alias="splitOn"),
    columns_metadatas: ColumnMetadataFromJSONStr = Form(
        None, alias="columnsMetadata"
    ),
    file: UploadFile = File(None),
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Create a dataset
    """
    try:
        payload = DatasetCreate(
            name=name,
            description=description,
            split_target=split_target,
            split_type=split_type,
            file=file,
            columns_metadata=columns_metadatas.metadatas,
        )

        if split_column:
            payload.split_column = split_column

        db_dataset = await controller.create_dataset(db, current_user, payload)

        dataset = Dataset.from_orm(db_dataset)
        return dataset

    except DatasetNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc

    except DatasetAlreadyExists as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Dataset name already in use",
        ) from exc


@REQUEST_TIME.labels(endpoint="/datasets/{dataset_id}", method="PUT").time()
@router.put(
    "/{dataset_id}",
    response_model=Dataset,
    dependencies=[Depends(deps.get_current_active_user)],
)
async def update_dataset(
    dataset_id: int,
    dataset_input: DatasetUpdateInput,
    current_user=Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
):
    """
    Update a dataset and process again if it is needed
    """
    try:
        dataset = await controller.update_dataset(
            db,
            current_user,
            dataset_id,
            DatasetUpdate(
                file=dataset_input.file,
                name=dataset_input.name,
                description=dataset_input.description,
                split_column=dataset_input.split_column,
                split_target=dataset_input.split_target,
                split_type=dataset_input.split_type,
                columns_metadata=dataset_input.columns_metadata.metadatas
                if dataset_input.columns_metadata
                else None,
            ),
        )
        return dataset
    except DatasetNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc
    except NotCreatorOwner as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc


@REQUEST_TIME.labels(endpoint="/datasets/{dataset_id}", method="DELETE").time()
@router.delete("/{dataset_id}", response_model=Dataset)
def delete_dataset(
    dataset_id: int,
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
):
    """Delete a dataset by id"""
    dataset = controller.delete_dataset(db, current_user, dataset_id)
    return dataset


@REQUEST_TIME.labels(endpoint="/datasets/csv-metadata", method="GET").time()
@router.post(
    "/csv-metadata",
    response_model=List[ColumnsMeta],
    dependencies=[Depends(deps.get_current_active_user)],
)
async def get_dataset_columns_metadata(file: UploadFile = File(None)):
    """Extracts column metadata for a given csv file"""
    metadata = await controller.parse_csv_headers(file)
    return metadata


@REQUEST_TIME.labels(
    endpoint="/datasets/{dataset_id}/file", method="GET"
).time()
@router.get(
    "/{dataset_id}/file",
    response_class=StreamingResponse,
    dependencies=[Depends(deps.get_current_active_user)],
)
def get_csv_file(
    dataset_id: int,
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
) -> StreamingResponse:
    """Get csv file for a given dataset id"""
    try:
        file = controller.get_csv_file(
            db, dataset_id, current_user, "original"
        )
        return StreamingResponse(file, media_type="text/csv")

    except DatasetNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc
    except NotCreatorOwner as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc


@REQUEST_TIME.labels(
    endpoint="/datasets/{dataset_id}/file-with-errors", method="GET"
).time()
@router.get(
    "/{dataset_id}/file-with-errors",
    response_class=StreamingResponse,
    dependencies=[Depends(deps.get_current_active_user)],
)
def get_csv_file_with_errors(
    dataset_id: int,
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
) -> StreamingResponse:
    """Get csv file for a given dataset id"""
    try:
        file = controller.get_csv_file(db, dataset_id, current_user, "error")
        return StreamingResponse(file, media_type="text/csv")

    except DatasetNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc
    except NotCreatorOwner as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc
