from typing import Any, Dict, NewType, Optional

import pandas as pd
import ray
from sqlalchemy.orm import Session
from torch_geometric.loader import DataLoader

from mariner.core import mlflowapi
from mariner.db.session import SessionLocal
from mariner.entities.deployment import DeploymentStatus
from mariner.exceptions import (
    DeploymentNotRunning,
    InvalidDataframe,
    ModelVersionNotFound,
    ModelVersionNotTrained,
    NotCreatorOwner,
)
from mariner.models import (
    CustomDataset,
    CustomModel,
    _check_dataframe_conforms_dataset,
)
from mariner.schemas.deployment_schemas import Deployment, DeploymentUpdateRepo
from mariner.schemas.model_schemas import ModelVersion
from mariner.stores.deployment_sql import deployment_store
from mariner.stores.model_sql import model_store
from model_builder.dataset import CustomDataset

db: Session = SessionLocal()


def get_db() -> Session:
    global db
    if not db:
        db = SessionLocal()
    return db


class DeploymentInstance:
    deployment: Deployment
    modelversion: ModelVersion
    is_running: bool
    model: Optional[CustomModel]
    db: Session = get_db()

    def __init__(self, deployment: Deployment):
        self.deployment = deployment
        self.update_status(DeploymentStatus.IDLE)

        self.modelversion = ModelVersion.from_orm(
            model_store.get_model_version(db, deployment.model_version_id)
        )
        if not self.modelversion:
            raise ModelVersionNotFound()
        if not self.modelversion.mlflow_version:
            raise ModelVersionNotTrained()

        self.is_running = False
        self.model = None

    def start(self):
        self.update_status(DeploymentStatus.STARTING)
        self.model = mlflowapi.get_model_by_uri(self.modelversion.get_mlflow_uri())
        self.is_running = True
        self.update_status(DeploymentStatus.ACTIVE)
        return self.deployment

    def stop(self):
        self.model = None
        self.is_running = False
        self.update_status(DeploymentStatus.IDLE)
        return self.deployment

    def predict(self, x: Any):
        if not self.is_running:
            raise DeploymentNotRunning("Deployment is not running")
        input_ = self.load_input(x)
        return self.model.predict_step(input_)

    def load_input(self, x):
        df = pd.DataFrame.from_dict(x, dtype=float)
        broken_checks = _check_dataframe_conforms_dataset(
            df, self.modelversion.config.dataset
        )

        if len(broken_checks) > 0:
            raise InvalidDataframe(
                f"dataframe failed {len(broken_checks)} checks",
                reasons=[f"{col_name}: {rule}" for col_name, rule in broken_checks],
            )

        dataset = CustomDataset(data=df, config=self.modelversion.config, target=False)
        dataloader = DataLoader(dataset, batch_size=len(df))
        return next(iter(dataloader))

    def update_status(self, status: DeploymentStatus):
        if not self.deployment:
            return

        self.deployment = deployment_store.update(
            self.db, self.deployment, DeploymentUpdateRepo(status=status.value)
        )
        # TODO: notify user by websocket


@ray.remote
class DeploymentsManager:
    deployments: Dict[int, DeploymentInstance]

    def __init__(self):
        self.deployments = {}

    def get_deployments(self):
        return self.deployments

    def add_deployment(self, deployment: Deployment):
        if not deployment.id in self.deployments.keys():
            self.deployments[deployment.id] = DeploymentInstance(deployment)
        return self.deployments[deployment.id].deployment

    def remove_deployment(self, deployment_id: int):
        if not deployment_id in self.deployments.keys():
            raise DeploymentNotRunning(
                "Deployment must be either running or idle to be removed"
            )
        self.deployments[deployment_id].update_status(DeploymentStatus.STOPPED)
        del self.deployments[deployment_id]

    def start_deployment(self, deployment_id: int, user_id: int):
        if not deployment_id in self.deployments.keys():
            raise DeploymentNotRunning(
                "Deployment must be either stopped or idle to be started"
            )

        if self.deployments[deployment_id].is_running:
            return self.deployments[deployment_id].deployment

        if self.deployments[deployment_id].deployment.created_by_id != user_id:
            raise NotCreatorOwner("Only the creator of the deployment can start it")

        return self.deployments[deployment_id].start()

    def stop_deployment(self, deployment_id: int, user_id: int):
        if not deployment_id in self.deployments.keys():
            raise DeploymentNotRunning(
                "Deployment must be either running or idle to be stopped"
            )

        if self.deployments[deployment_id].deployment.created_by_id != user_id:
            raise NotCreatorOwner("Only the creator of the deployment can stop it")

        return self.deployments[deployment_id].stop()

    def make_prediction(self, deployment_id: int, x: Any):
        if (
            not self.deployments[deployment_id]
            or not self.deployments[deployment_id].is_running
        ):
            raise DeploymentNotRunning("Deployment must be running to make predictions")

        return self.deployments[deployment_id].predict(x)


manager = None
Manager = NewType("Remote Deployment Manager", DeploymentsManager)


def get_deployments_manager() -> Manager:
    """Make sure that only one instance of the manager is created"""
    global manager
    if not manager:
        manager = DeploymentsManager.remote()
    return manager
