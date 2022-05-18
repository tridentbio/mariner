from typing import Any, Generic, List, TypeVar

from fastapi.datastructures import UploadFile
from fastapi.param_functions import Depends, File, Form
from fastapi.routing import APIRouter
from pydantic.generics import GenericModel
from sqlalchemy.orm.session import Session

from app.api import deps
from app.features.dataset import controller
from app.features.dataset.schema import (
    Dataset,
    DatasetCreate,
    DatasetsQuery,
    Split,
    SplitType,
)

router = APIRouter()

# TODO: move to utils
DataT = TypeVar("DataT")


class Paginated(GenericModel, Generic[DataT]):
    data: List[DataT]
    total: int


@router.get("/", response_model=Paginated[Dataset])
def get_my_datasets(
    query=Depends(DatasetsQuery),
    current_user=Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Retrieve datasets owned by requester
    """
    datasets, total = controller.get_my_datasets(db, current_user, query)
    return Paginated(data=[Dataset.from_orm(ds) for ds in datasets], total=total)


@router.post("/", response_model=Dataset)
def create_dataset(
    current_user=Depends(deps.get_current_active_user),
    name: str = Form(...),
    description: str = Form(...),
    split_target: Split = Form(...),
    split_type: SplitType = Form(...),
    file: UploadFile = File(None),
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Create a dataset
    """
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
