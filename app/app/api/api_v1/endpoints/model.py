from typing import Optional

from fastapi.datastructures import UploadFile
from fastapi.param_functions import Depends, File, Form
from fastapi.routing import APIRouter
from sqlalchemy.orm.session import Session

from app.api import deps
from app.api.api_v1.endpoints.datasets import Paginated
from app.features.model import controller
from app.features.model.schema.model import Model, ModelCreate, ModelsQuery
from app.features.user.model import User

router = APIRouter()


@router.post(
    "/",
    response_model=Model,
)
def create_model(
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_user),
    name: str = Form(...),
    file: UploadFile = File(None),
    description: Optional[str] = Form(None),
    version_description: Optional[str] = Form(None),
) -> Model:
    model = controller.create_model(
        db,
        ModelCreate(
            name=name,
            created_by_id=current_user.id,
            model_description=description,
            model_version_description=version_description,
        ),
        file,
    )
    return model

@router.get(
    "/",
    response_model=Paginated[Model],
)
def get_models(
    db: Session = Depends(deps.get_db),
    query: ModelsQuery = Depends(ModelsQuery),
    current_user: User = Depends(deps.get_current_active_user)
):
    models, total = controller.get_models(db, query, current_user=current_user)
    models = [ Model.from_orm(m) for m in models ]
    return Paginated(data=models, total=total)

