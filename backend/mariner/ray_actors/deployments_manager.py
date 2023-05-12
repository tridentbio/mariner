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
    PredictionLimitReached,
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


def update_deployment_status(
    deployment: Deployment, status: DeploymentStatus
) -> Deployment:
    global db
    deployment = deployment_store.update(
        db, deployment, DeploymentUpdateRepo(status=status.value)
    )
    return deployment


class ModelManager:
    deployment: Deployment
    modelversion: ModelVersion
    is_running: bool
    model: Optional[CustomModel]

    def __init__(self, deployment: Deployment):
        self.deployment = update_deployment_status(deployment, DeploymentStatus.IDLE)

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
        self.deployment = update_deployment_status(
            self.deployment, DeploymentStatus.STARTING
        )
        yield self.deployment

        self.model = mlflowapi.get_model_by_uri(self.modelversion.get_mlflow_uri())
        self.is_running = True
        self.deployment = update_deployment_status(
            self.deployment, DeploymentStatus.ACTIVE
        )
        # TODO: Send message to user that deployment is ready

    def stop(self):
        self.model = None
        self.is_running = False
        self.deployment = update_deployment_status(
            self.deployment, DeploymentStatus.IDLE
        )
        return self.deployment

    def predict(self, x: Any):
        if not self.is_running:
            raise DeploymentNotRunning("Deployment is not running")
        input_ = self.load_input(x)
        return self.model(input_)

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

    def __del__(self):
        if self.is_running:
            self.stop()
        update_deployment_status(self.deployment, DeploymentStatus.STOPPED)


@ray.remote
class DeploymentsManager:
    deployments: Dict[str, ModelManager]

    def __init__(self):
        self.deployments = {}

    def add_deployment(self, deployment: Deployment):
        if not deployment.id in self.deployments.keys():
            self.deployments[deployment.id] = ModelManager(deployment)
        return self.deployments[deployment.id]

    def remove_deployment(self, deployment_id: int):
        if not deployment_id in self.deployments.keys():
            raise DeploymentNotRunning(
                "Deployment must be either running or idle to be removed"
            )
        del self.deployments[deployment_id]

    def start_deployment(self, deployment_id: int, user_id: int):
        if not deployment_id in self.deployments.keys():
            raise DeploymentNotRunning(
                "Deployment must be either running or idle to be started"
            )

        if self.deployments[deployment_id].is_running:
            return self.deployments[deployment_id].deployment

        if self.deployments[deployment_id].deployment.created_by_id != user_id:
            raise NotCreatorOwner("Only the creator of the deployment can start it")

        return next(self.deployments[deployment_id].start())

    def stop_deployment(self, deployment_id: int, user_id: int):
        if not deployment_id in self.deployments.keys():
            raise DeploymentNotRunning(
                "Deployment must be either running or idle to be stopped"
            )

        if not self.deployments[deployment_id].is_running:
            return self.deployments[deployment_id].deployment

        if self.deployments[deployment_id].deployment.created_by_id != user_id:
            raise NotCreatorOwner("Only the creator of the deployment can stop it")

        return self.deployments[deployment_id].stop()

    def check_prediction_limit(self, deployment_id: int, user_id: int):
        return False

    def track_prediction(self, deployment_id: int, user_id: int):
        # TODO: track prediction on db
        ...

    def make_prediction(self, deployment_id: int, user_id: int, x: Any):
        if self.check_prediction_limit(deployment_id, user_id):
            raise PredictionLimitReached("Prediction limit reached")

        if (
            not self.deployments[deployment_id]
            or not self.deployments[deployment_id].is_running
        ):
            raise DeploymentNotRunning("Deployment must be running to make predictions")

        prediction = self.deployments[deployment_id].predict(x)
        self.track_prediction(deployment_id, user_id)

        return prediction

    def __del__(self):
        for model_manager in self.deployments.values:
            del model_manager


manager = DeploymentsManager.remote()
Manager = NewType("Remote Deployment Manager", DeploymentsManager)


def get_deployments_manager() -> Manager:
    """Make sure that only one instance of the manager is created"""
    global manager
    if not manager:
        manager = DeploymentsManager.remote()
    return manager
