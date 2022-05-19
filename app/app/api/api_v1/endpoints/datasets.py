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
from app.features.dataset.exceptions import DatasetNotFound, NotCreatorOfDataset
from app.features.dataset.schema import (
    Dataset,
    DatasetCreate,
    DatasetsQuery,
    DatasetUpdate,
    Split,
    SplitType,
)
from app.features.user.model import User

router = APIRouter()

# TODO: move to utils
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


@router.post("/", response_model=Dataset)
def create_dataset(
    current_user: User = Depends(deps.get_current_active_user),
    name: str = Form(...),
    description: str = Form(...),
    split_target: Split = Form(..., alias="splitTarget"),
    split_type: SplitType = Form(..., alias="splitType"),
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
        )
        db_dataset = controller.create_dataset(db, current_user, payload)
        dataset = Dataset.from_orm(db_dataset)
        return dataset
    except DatasetNotFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)


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
