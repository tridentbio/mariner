from typing import List

from fastapi.param_functions import Depends
from fastapi.routing import APIRouter
from sqlalchemy.orm.session import Session

from app.api import deps
from app.features.experiments import controller as experiments_ctl
from app.features.experiments.schema import (
    Experiment,
    ListExperimentsQuery,
    RunningHistory,
    TrainingRequest,
)
from app.features.user.model import User

router = APIRouter()


@router.post("/", response_model=Experiment)
async def post_experiments(
    request: TrainingRequest,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_user),
) -> Experiment:
    result = await experiments_ctl.create_model_traning(db, current_user, request)
    return result


@router.get("/", response_model=List[Experiment])
def get_experiments(
    experiments_query: ListExperimentsQuery = Depends(),
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_user),
) -> List[Experiment]:
    result = experiments_ctl.get_experiments(db, current_user, experiments_query)
    return result

@router.get("/running-history", response_model=List[RunningHistory])
def get_experiments_running_history(
    user: User = Depends(deps.get_current_active_user),
) -> List[RunningHistory]:
    histories = experiments_ctl.get_running_histories(user)
    return histories

