from fastapi.param_functions import Depends
from fastapi.routing import APIRouter
from sqlalchemy.orm.session import Session

from app.api import deps
from app.features.model import controller
from app.features.model.schema.model import Experiment, TrainingRequest
from app.features.user.model import User

router = APIRouter()


@router.post("/", response_model=Experiment)
def post_experiments(
    request: TrainingRequest,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_user),
):
    result = controller.create_model_traning(db, current_user, request)
    return result
