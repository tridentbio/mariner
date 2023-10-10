"""
Handlers for api/v1/models* endpoints
"""
from typing import Annotated, Any, Dict, List

from fastapi import Body
from fastapi.exceptions import HTTPException
from fastapi.param_functions import Depends
from fastapi.routing import APIRouter
from sqlalchemy.orm.session import Session
from starlette import status

import mariner.models as controller
from api import deps
from api.api_v1.endpoints.datasets import Paginated
from fleet.dataset_schemas import AllowedLosses
from fleet.options import ComponentOption
from mariner.entities.user import User
from mariner.exceptions import DatasetNotFound, ModelNameAlreadyUsed
from mariner.exceptions.model_exceptions import (
    InvalidDataframe,
    ModelNotFound,
    ModelVersionNotTrained,
)
from mariner.schemas.api import ApiBaseModel
from mariner.schemas.model_schemas import Model, ModelCreate, ModelsQuery
from mariner.utils import random_pretty_name

router = APIRouter()


@router.post(
    "/",
    response_model=Model,
)
async def create_model(
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
        model = await controller.create_model(
            db,
            current_user,
            model_create,
        )
        return model
    except ModelNameAlreadyUsed as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Another model is already registered with that name",
        ) from exc
    except DatasetNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'Dataset "{model_create.config.dataset.name}" not found',
        ) from exc


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
    response_model=List[ComponentOption],
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


@router.post(
    "/{model_version_id}/predict",
    response_model=Dict[str, List[Any]],
)
def post_model_predict(
    model_version_id: int,
    model_input: Annotated[
        Dict[str, List[Any]],
        Body(
            examples=[
                {
                    "smiles": ["CC", "CCC"],
                    "mwt": [16.0, 30.0],
                }
            ]
        ),
    ],  # Any json
    current_user: User = Depends(deps.get_current_active_user),
    db: Session = Depends(deps.get_db),
) -> Any:
    """Gets a prediction from a trained model.

    The request body must be a json from which we can bulid the input matrix.
    E.g. You've trained a model to predict the ``tpsa`` property from the
    ``smiles`` and ``mwt`` properties. The body would be:

    ```json
    {
        "smiles": ["CC", "CCC"],
        "mwt": [16.0, 30.0],
    }
    ```

    And the returned prediction could be:

    ```json
    {
        "tpsa": [12.0, 20.0],
    }
    ```
    """
    try:
        result = controller.get_model_prediction(
            db,
            controller.PredictRequest(
                user_id=current_user.id,
                model_version_id=model_version_id,
                model_input=model_input,
            ),
        )
        return result
    except InvalidDataframe as exp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Payload failed following checks:{','.join(exp.reasons)}",
        ) from exp
    except ModelNotFound as exp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Model Not Found"
        ) from exp
    except ModelVersionNotTrained as exp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model version was not trained yet",
        ) from exp


@router.get(
    "/losses",
    response_model=AllowedLosses,
)
def get_model_losses():
    """Gets the available loss functions that can be used for training neural
    networks."""
    return AllowedLosses()


@router.get("/{model_id}", response_model=Model)
def get_model(
    model_id: int,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_user),
):
    """Gets a single model from the database"""
    model = controller.get_model(db, current_user, model_id)
    return model


@router.delete("/{model_id}", response_model=Model)
def delete_model(
    model_id: int,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_user),
):
    """Deletes a model from the database and mlflow."""
    model = controller.delete_model(db, current_user, model_id)
    return model
