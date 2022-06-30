from fastapi.param_functions import Depends
from fastapi.routing import APIRouter
from sqlalchemy.orm.session import Session

from app.api import deps
from app.features.model import controller
from app.features.model.deployments.schema import Deployment, DeploymentCreate

router = APIRouter()


@router.post(
    "/",
    response_model=Deployment,
    dependencies=[Depends(deps.get_current_active_user)],
)
def create_deployment(
    deployment_create: DeploymentCreate,
    db: Session = Depends(deps.get_db),
    current_user=Depends(deps.get_current_active_user),
):
    print(deployment_create, current_user)
    deployment = controller.create_model_deployment(db, deployment_create, current_user)
    return deployment
