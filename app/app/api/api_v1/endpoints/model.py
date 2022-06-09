from typing import Optional

from fastapi.datastructures import UploadFile
from fastapi.param_functions import Depends, File, Form
from fastapi.routing import APIRouter
from sqlalchemy.orm.session import Session

from app.api import deps
from app.features.model import controller
from app.features.model.schema.model import Model, ModelCreate

router = APIRouter()


@router.post(
    "/models/",
    response_model=Model,
    dependencies=[Depends(deps.get_current_active_user)],
)
def create_model(
    db: Session = Depends(deps.get_db),
    name: str = Form(...),
    file: UploadFile = File(None),
    description: Optional[str] = Form(None),
    version_description: Optional[str] = Form(None),
) -> Model:
    model = controller.create_model(
        db,
        ModelCreate(
            name=name,
            model_description=description,
            model_version_description=version_description,
        ),
        file,
    )
    return model
