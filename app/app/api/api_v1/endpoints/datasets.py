from typing import Any, Generic, List, Optional, TypeVar

from fastapi.datastructures import UploadFile
from fastapi.exceptions import HTTPException
from fastapi.param_functions import Depends, File, Form
from fastapi.routing import APIRouter
from pydantic.generics import GenericModel
from sqlalchemy.orm.session import Session
from starlette import status

from app.api import deps
from app.features.dataset import controller
from app.features.dataset.exceptions import (
    DatasetAlreadyExists,
    DatasetNotFound,
    NotCreatorOfDataset,
)
from app.features.dataset.schema import (
    ColumnMetadataFromJSONStr,
    ColumnsMeta,
    Dataset,
    DatasetCreate,
    DatasetsQuery,
    DatasetUpdate,
    Split,
    SplitType,
)
from app.features.user.model import User

router = APIRouter()

DataT = TypeVar("DataT")


class Paginated(GenericModel, Generic[DataT]):
    data: List[DataT]
    total: int


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
    return Paginated(data=[Dataset.from_orm(ds) for ds in datasets], total=total)


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


@router.post("/", response_model=Dataset)
def create_dataset(
    current_user: User = Depends(deps.get_current_active_user),
    name: str = Form(...),
    description: str = Form(...),
    split_target: Split = Form(..., alias="splitTarget"),
    split_type: SplitType = Form(..., alias="splitType"),
    columns_metadatas: ColumnMetadataFromJSONStr = Form(None, alias="columnsMetadata"),
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
        db_dataset = controller.create_dataset(db, current_user, payload)
        dataset = Dataset.from_orm(db_dataset)
        return dataset
    except DatasetNotFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    except DatasetAlreadyExists:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Dataset name already in use"
        )


@router.put(
    "/{dataset_id}",
    response_model=Dataset,
    dependencies=[Depends(deps.get_current_active_user)],
)
def update_dateset(
    dataset_id: int,
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    split_target: Optional[Split] = Form(None, alias="splitTarget"),
    split_type: Optional[SplitType] = Form(
        None,
        alias="splitType",
    ),
    file: Optional[UploadFile] = File(None),
    current_user=Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
):
    try:
        dataset = controller.update_dataset(
            db,
            current_user,
            dataset_id,
            DatasetUpdate(
                file=file,
                name=name,
                description=description,
                split_target=split_target,
                split_type=split_type,
            ),
        )
        return dataset
    except DatasetNotFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    except NotCreatorOfDataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)


@router.delete("/{dataset_id}", response_model=Dataset)
def delete_dataset(
    dataset_id: int,
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
):
    dataset = controller.delete_dataset(db, current_user, dataset_id)
    return dataset


@router.post(
    "/csv-metadata",
    response_model=List[ColumnsMeta],
    dependencies=[Depends(deps.get_current_active_user)],
)
def get_columns_metadata(file: UploadFile = File(None)):
    """Extracts column metadata for a given csv file"""
    metadata = controller.parse_csv_headers(file)
    return metadata
