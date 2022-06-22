from app.crud.base import CRUDBase
from app.features.model.deployments.model import Deployment
from app.features.model.deployments.schema import (
    DeploymentCreate,
    DeploymentUpdate,
)


class CRUDDeployment(CRUDBase[Deployment, DeploymentCreate, DeploymentUpdate]):
    pass


repo = CRUDDeployment(Deployment)
