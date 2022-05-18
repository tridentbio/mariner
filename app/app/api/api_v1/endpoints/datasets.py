from typing import Any, Generic, List, TypeVar
from fastapi.datastructures import UploadFile
from pydantic.generics import GenericModel
from fastapi.param_functions import Depends, File
from fastapi.routing import APIRouter
from sqlalchemy.orm.session import Session

from app.api import deps
from app.features.dataset.schema import DatasetCreate, DatasetsQuery, Dataset
from app.features.dataset import controller


router = APIRouter()

# TODO: move to utils
DataT = TypeVar('DataT')
class Paginated (GenericModel, Generic[DataT]):
    data: List[DataT]
    total: int

@router.get('/', response_model=Paginated[Dataset])
def get_my_datasets(
    query = Depends(DatasetsQuery),
    current_user = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db)
) -> Any:
    """
    Retrieve datasets owned by requester
    """
    print(current_user, query)
    datasets, total = controller.get_my_datasets(db, current_user, query)
    return Paginated(data=datasets, total=total)

@router.post('/', response_model=Dataset)
def create_dataset(
    file: UploadFile = File(None),
    payload = Depends(DatasetCreate),
    current_user = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db)
    ) -> Any:
    """
    Create a dataset
    """
    db_dataset = controller.create_dataset(db, current_user, file, payload)
    dataset = Dataset.from_orm(db_dataset)
    return dataset


