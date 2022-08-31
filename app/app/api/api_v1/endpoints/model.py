from typing import Any, Dict, List

import torch
from fastapi.exceptions import HTTPException
from fastapi.param_functions import Depends
from fastapi.routing import APIRouter
from pandas.core.frame import DataFrame
from sqlalchemy.orm.session import Session
from starlette import status

from app.api import deps
from app.api.api_v1.endpoints.datasets import Paginated
from app.features.dataset.exceptions import DatasetNotFound
from app.features.model import controller
from app.features.model.exceptions import ModelNameAlreadyUsed, ModelNotFound
from app.features.model.schema.configs import ModelConfig, ModelOptions
from app.features.model.schema.model import Model, ModelCreate, ModelsQuery
from app.features.user.model import User

router = APIRouter()


@router.post(
    "/",
    response_model=Model,
)
def create_model(
    model_create: ModelCreate,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_user),
) -> Model:
    try:
        model = controller.create_model(
            db,
            current_user,
            model_create,
        )
        return model
    except ModelNameAlreadyUsed:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Another model is already registered with that name",
        )
    except DatasetNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'Dataset "{model_create.config.dataset.name}" not found',
        )


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


@router.post("/{model_version_id}/predict", response_model=Any)
def post_predict(
    model_version_id: int,
    model_input: Dict[str, List[Any]],  # Any json
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
) -> Any:
    try:
        prediction = controller.get_model_prediction(
            db,
            controller.PredictRequest(
                user_id=current_user.id,
                model_version_id=model_version_id,
                model_input=model_input,
            ),
        )
    except ModelNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Model Not Found"
        )
    if isinstance(prediction, torch.Tensor):
        return prediction.tolist()
    elif isinstance(prediction, DataFrame):
        return prediction.to_json()
    else:
        raise TypeError("Unexpected model output")


@router.get("/{model_id}", response_model=Model)
def get_model(
    model_id: int,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_user),
):
    model = controller.get_model_version(db, current_user, model_id)
    return model


@router.delete("/{model_id}", response_model=Model)
def delete_model(
    model_id: int,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_user),
):
    model = controller.delete_model(db, current_user, model_id)
    return model


@router.post(
    "/check-config",
    response_model=controller.ForwardCheck,
    dependencies=[Depends(deps.get_current_active_user)],
)
def post_check_config(model_config: ModelConfig, db: Session = Depends(deps.get_db)):
    return controller.naive_check_forward_exception(db, model_config)
