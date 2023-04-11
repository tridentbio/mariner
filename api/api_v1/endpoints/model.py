"""
Handlers for api/v1/models* endpoints
"""
from typing import Any, Dict, List, Optional, Union

import torch
from fastapi.exceptions import HTTPException
from fastapi.param_functions import Depends
from fastapi.routing import APIRouter
from sqlalchemy.orm.session import Session
from starlette import status

import mariner.models as controller
from api import deps
from api.api_v1.endpoints.datasets import Paginated
from mariner.entities.user import User
from mariner.exceptions import (
    DatasetNotFound,
    ModelNameAlreadyUsed,
    ModelNotFound,
)
from mariner.exceptions.model_exceptions import (
    InvalidDataframe,
    ModelVersionNotTrained,
)
from mariner.schemas.api import ApiBaseModel
from mariner.schemas.model_schemas import (
    Model,
    ModelCreate,
    ModelOptions,
    ModelSchema,
    ModelsQuery,
)
from mariner.utils import random_pretty_name
from model_builder.schemas import AllowedLosses

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
    """Creates a model and model version, or adds a model version to existing model.

    Creates a model and model version in the database and in mlflow. The entities from
    mlflow are mapped from the Mariner entities through mlflow* columns, like
    mlflow_name and mlflow_id. When the payload refers to an already existing model,
    than a model version is added to it.

    Args:
        model_create: Payload specifying model.
        db: Connection to the database.
        current_user: User that originated the request.

    Returns:
        The created model.

    Raises:
        HTTPException: 409 when model name conflicts.
        HTTPException: 404 when some reference in payload doesn't exist.
    """
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
    """Gets a page of models.

    Args:
        db: Connection to database.
        query: Query specifying what models to get.
        current_user: The user that originated the request.
    """
    models, total = controller.get_models(db, query, current_user=current_user)
    models = [Model.from_orm(m) for m in models]
    return Paginated(data=models, total=total)


@router.get(
    "/options",
    response_model=ModelOptions,
)
def get_model_options():
    """Gets the model building options and documentations."""
    model_options = controller.get_model_options()
    return model_options


class GetNameSuggestionResponse(ApiBaseModel):
    """Payload for a name suggestion.

    Attributes:
        name: the name suggested.
    """

    name: str


@router.get(
    "/name-suggestion",
    dependencies=[Depends(deps.get_current_active_user)],
    response_model=GetNameSuggestionResponse,
)
def get_model_name_suggestion():
    """Endpoint to get a name suggestion."""
    return GetNameSuggestionResponse(name=random_pretty_name())


@router.post("/{model_version_id}/predict", response_model=Dict[str, List[Any]])
def post_model_predict(
    model_version_id: int,
    model_input: Dict[str, List[Any]],  # Any json
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
) -> Any:
    """Endpoint to use a trained model for prediction.

    Args:
        model_version_id: Id of the model version to be used for prediction.
        model_input: JSON version of the pandas dataframe used as input.
        current_user: User that originated the request.
        db: Connection to the db

    Returns:
        The model prediction as a torch tensor or pandas dataframe.

    Raises:
        HTTPException: When app is not able to get the prediction
        TypeError: When the model output is not a Tensor or a Dataframe
    """
    prediction: Optional[Dict[str, Union[torch.Tensor, List[Any]]]] = None
    try:
        prediction = controller.get_model_prediction(
            db,
            controller.PredictRequest(
                user_id=current_user.id,
                model_version_id=model_version_id,
                model_input=model_input,
            ),
        )
    except InvalidDataframe as exp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Payload failed following checks:{','.join(exp.reasons)}",
        )
    except ModelNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Model Not Found"
        )
    except ModelVersionNotTrained:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model version was not trained yet",
        )

    for column, result in prediction.items():
        if not isinstance(result, torch.Tensor):
            raise TypeError("Unexpected model output")
        serialized_result = result.tolist()
        prediction[column] = (
            serialized_result
            if isinstance(serialized_result, list)
            else [serialized_result]
        )

    return prediction


@router.get(
    "/losses",
    response_model=AllowedLosses,
)
def get_model_losses():
    """Endpoint to get the available losses"""
    return AllowedLosses()


@router.get("/{model_id}", response_model=Model)
def get_model(
    model_id: int,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_user),
):
    """Endpoint to get a single model from the database.

    Args:
        model_id: Id of the model.
        db: Database connection.
        current_user: User that is requested the action
    """
    model = controller.get_model(db, current_user, model_id)
    return model


@router.delete("/{model_id}", response_model=Model)
def delete_model(
    model_id: int,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_user),
):
    """Endpoint to delete a model

    Args:
        model_id: id of the model to be deleted
        db: Connection to the database.
        current_user: User that is requested the action
    """
    model = controller.delete_model(db, current_user, model_id)
    return model


@router.post(
    "/check-config",
    response_model=controller.ForwardCheck,
    dependencies=[Depends(deps.get_current_active_user)],
)
async def post_model_check_config(
    model_config: ModelSchema, db: Session = Depends(deps.get_db)
):
    """Endpoint to check the forward method of a ModelSchema

    Args:
        model_config: Model schema to be checked
        db: Database connection
    """
    result = await controller.check_model_step_exception(db, model_config)
    return result
