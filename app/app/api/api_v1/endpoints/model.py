from typing import Any, Optional

from fastapi.datastructures import UploadFile
from fastapi.exceptions import HTTPException
from fastapi.param_functions import Depends, File, Form
from fastapi.routing import APIRouter
from pandas.core.frame import DataFrame
from pydantic.main import BaseModel
from sqlalchemy.orm.session import Session
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_404_NOT_FOUND

from app.api import deps
from app.api.api_v1.endpoints.datasets import Paginated
from app.features.model import controller
from app.features.model.exceptions import ModelNotFound
from app.features.model.schema.configs import ModelOptions
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
    current_user: User = Depends(deps.get_current_active_user),
):
    models, total = controller.get_models(db, query, current_user=current_user)
    models = [Model.from_orm(m) for m in models]
    return Paginated(data=models, total=total)


@router.get(
    "/options",
    response_model=ModelOptions,
)
def get_model_options():
    model_options = controller.get_model_options()
    return model_options


@router.post("/{user_id}/{model_name}/{model_version}/predict", response_model=Any)
def post_predict(
    model_name: str,
    user_id: int,
    model_version: str,
    model_input=Depends(BaseModel),  # Any json
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
) -> Any:
    if user_id != current_user.id:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED)
    try:
        prediction = controller.get_model_prediction(
            db,
            controller.PredictRequest(
                user_id=current_user.id,
                model_name=model_name,
                model_input=model_input,
                version=model_version,
            ),
        )
    except ModelNotFound:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Model Not Found")
    assert isinstance(prediction, DataFrame)
    return prediction.to_json()
