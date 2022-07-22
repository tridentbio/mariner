from typing import List
from fastapi.param_functions import Depends
from fastapi.routing import APIRouter
from sqlalchemy.orm.session import Session

from app.api import deps
from app.features.experiments import controller as experiments_ctl
from app.features.experiments.schema import ListExperimentsQuery
from app.features.experiments.schema import Experiment, TrainingRequest
from app.features.user.model import User

router = APIRouter()


@router.post("/", response_model=Experiment)
async def post_experiments(
    request: TrainingRequest,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_user),
):
    result = await experiments_ctl.create_model_traning(db, current_user, request)
    return result

@router.get("/", response_model=List[Experiment])
def get_experiments(
    experiments_query: ListExperimentsQuery = Depends(),
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    result = experiments_ctl.get_experiments(db, current_user, experiments_query)
    return result

