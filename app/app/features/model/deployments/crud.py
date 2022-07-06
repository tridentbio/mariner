from app.crud.base import CRUDBase
from app.features.model.deployments.model import Deployment
from app.features.model.deployments.schema import (
    DeploymentCreateRepo,
    DeploymentUpdate,
)


class CRUDDeployment(CRUDBase[Deployment, DeploymentCreateRepo, DeploymentUpdate]):
    pass


repo = CRUDDeployment(Deployment)
