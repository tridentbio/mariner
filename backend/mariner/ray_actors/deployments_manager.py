from threading import Thread
from time import sleep, time
from typing import Any, Dict, List, NewType, Optional

import pandas as pd
import ray
import requests
from fleet.model_builder.dataset import CustomDataset
from fleet.torch_.models import CustomModel
from mariner.core import mlflowapi
from mariner.core.config import settings
from mariner.entities.deployment import DeploymentStatus
from mariner.exceptions import (DeploymentNotRunning, InvalidDataframe,
                                ModelVersionNotFound, ModelVersionNotTrained,
                                NotCreatorOwner)
from mariner.models import _check_dataframe_conforms_dataset
from mariner.schemas.deployment_schemas import (Deployment,
                                                DeploymentManagerComunication)
from torch_geometric.loader import DataLoader


class SingleModelDeploymentControl:
    """Responsible for managing the deployment of a model.
    
    Used by the :class:`DeploymentsManager` to manage the lifecycle of a deployment.

    Possible status of a deployment:
    
    - IDLE: The deployment already have a :class:`SingleModelDeploymentControl` but it is not running.
    - STARTING: The deployment is starting.
    - ACTIVE: The deployment was loaded from mlflow and it's running.
    - STOPPED: The deployment don't have a :class:`SingleModelDeploymentControl`.
    """

    deployment: Deployment
    is_running: bool
    model: Optional[CustomModel]
    idle_time: Optional[int] = None

    def __init__(self, deployment: Deployment):
        self.deployment = deployment
        self.update_status(DeploymentStatus.IDLE)

        self.is_running = False
        self.model = None

    @property
    def idle_long(self) -> bool:
        """Checks if the deployment is idle for a long time.

        Returns:
            True if the deployment is idle for a long time, False otherwise.
        """
        return (
            self.idle_time and (time() - self.idle_time) > settings.DEPLOYMENT_IDLE_TIME
        )

    def start(self) -> Deployment:
        """Starts the SingleModelDeploymentControl, which includes loading the
        model and updating its status to ACTIVE.

        Returns:
            The updated deployment after starting the instance.
        """
        self.update_status(DeploymentStatus.STARTING)
        self.model = mlflowapi.get_model_by_uri(
            self.deployment.model_version.get_mlflow_uri()
        )
        self.is_running = True
        self.update_status(DeploymentStatus.ACTIVE)
        return self.deployment

    def stop(self) -> Deployment:
        """Stops the SingleModelDeploymentControl by setting ending its model
        cycle, updating its running status to False, and setting its
        status to IDLE.

        Returns:
            The updated deployment after stopping the instance.
        """
        self.model = None
        self.is_running = False
        self.update_status(DeploymentStatus.IDLE)
        return self.deployment

    def predict(self, x: Any) -> Any:
        """Given an input x, makes a prediction using the model.

        Args:
            x (Any): Input data to be used for generating predictions.

        Returns:
            The predicted output generated by the model.
        """
        if not self.is_running:
            raise DeploymentNotRunning("Deployment is not running")
        input_ = self.load_input(x)
        return self.model.predict_step(input_)

    def load_input(self, x):
        """Loads input data into a format compatible with the model
        for prediction.

        Args:
            x (Any): Input data to be converted.

        Returns:
            A DataLoader object containing input data in the required format.

        Raises:
            InvalidDataframe:
                If the input data does not conform to the model's dataset.
        """
        df = pd.DataFrame.from_dict(x, dtype=float)
        broken_checks = _check_dataframe_conforms_dataset(
            df, self.deployment.model_version.config.dataset
        )

        if len(broken_checks) > 0:
            raise InvalidDataframe(
                f"dataframe failed {len(broken_checks)} checks",
                reasons=[f"{col_name}: {rule}" for col_name, rule in broken_checks],
            )
        dataset = CustomDataset(
            data=df,
            model_config=self.deployment.model_version.config.spec,
            dataset_config=self.deployment.model_version.config.dataset,
            target=False,
        )
        dataloader = DataLoader(dataset, batch_size=len(df))
        return next(iter(dataloader))

    def update_status(self, status: DeploymentStatus):
        """Updates the deployment status and stores the updated
        status in the database.

        Args:
            status (DeploymentStatus): The new status for the deployment.
        """
        if status == DeploymentStatus.IDLE:
            self.idle_time = time()
        else:
            self.idle_time = None

        data = DeploymentManagerComunication(
            deployment_id=self.deployment.id, status=status
        )
        res = requests.post(
            f"{settings.SERVER_HOST}/api/v1/deployments/deployment-manager",
            json=data.dict(),
            headers={"Authorization": f"Bearer {settings.APPLICATION_SECRET}"},
        )
        assert (
            res.status_code == 200, 
            f"Request to update deployment failed with status  {res.status_code}"
        )

        self.deployment = Deployment(**res.json())


@ray.remote
class DeploymentsManager:
    """Ray Actor responsible for managing multiple instances
    of :class:`SingleModelDeploymentControl` and handling their life cycle
    (starting, stopping, etc.). Should be used as a singleton.
    """

    deployments: Dict[int, SingleModelDeploymentControl]

    def __init__(self):
        self.deployments = {}
        self.load()
        Thread(target=self.check_idle_long_loop).start()

    def get_deployments(self):
        """Returns a dictionary containing all the registered
        :class:`SingleModelDeploymentControl` objects.
        """
        return self.deployments

    def is_deployment_running(self, deployment_id: int) -> bool:
        """Checks if the specified deployment is running on
        current ray actor.

        Args:
            deployment_id (int): The ID of the deployment to check.

        Returns:
            True if the deployment is running, False otherwise.
        """
        return (
            deployment_id in self.deployments.keys()
            and self.deployments[deployment_id].is_running
        )

    def add_deployment(self, deployment: Deployment) -> Deployment:
        """Adds a new deployment to the DeploymentsManager.
        
        New deployment should be in the IDLE state when added.

        Args:
            deployment (Deployment): The deployment to be added.

        Returns:
            The added deployment.
        
        Raises:
            ModelVersionNotFound when the deployment does not have a model version.
            ModelVersionNotTrained when the deployment's model version is not trained yet.
        """
        if not deployment.model_version:
            raise ModelVersionNotFound()
        if not deployment.model_version.mlflow_version:
            raise ModelVersionNotTrained()
        
        if not deployment.id in self.deployments.keys():
            self.deployments[deployment.id] = SingleModelDeploymentControl(deployment)
        return self.deployments[deployment.id].deployment

    def remove_deployment(self, deployment_id: int):
        """Removes a deployment from the DeploymentsManager.
        
        A deployment should be in STOPPED state after removed.

        Args:
            deployment_id (int): The ID of the deployment to be removed.

        Raises:
            DeploymentNotRunning:
                If the deployment is not mapped to a :class:`SingleModelDeploymentControl`.
        """
        if not deployment_id in self.deployments.keys():
            raise DeploymentNotRunning(
                "Deployment must be either running or idle to be removed"
            )
        self.deployments[deployment_id].update_status(DeploymentStatus.STOPPED)
        del self.deployments[deployment_id]

    def start_deployment(self, deployment_id: int, user_id: int) -> Deployment:
        """Starts a deployment with the given deployment_id and user_id.
        
        The deployment should already be added to the DeploymentsManager
        before starting it.

        Args:
            deployment_id (int): The ID of the deployment to start.
            user_id (int): The ID of the user attempting to start the deployment.

        Returns:
            The updated deployment after it has started.

        Raises:
            DeploymentNotRunning:
                If the deployment is not mapped to a :class:`SingleModelDeploymentControl`.
            NotCreatorOwner:
                If the user attempting to start the deployment is not the creator.
        """
        if not deployment_id in self.deployments.keys():
            raise DeploymentNotRunning(
                "Deployment must be either stopped or idle to be started"
            )

        if self.deployments[deployment_id].is_running:
            return self.deployments[deployment_id].deployment

        if self.deployments[deployment_id].deployment.created_by_id != user_id:
            raise NotCreatorOwner("Only the creator of the deployment can start it")

        return self.deployments[deployment_id].start()

    def stop_deployment(self, deployment_id: int, user_id: int) -> Deployment:
        """Stops a deployment with the given deployment_id and user_id.
        
        A deployment should be in RUNNING state to be stopped.
        A deployment should be in IDLE state after stopped.

        Args:
            deployment_id (int): The ID of the deployment to stop.
            user_id (int): The ID of the user attempting to stop the deployment.

        Returns:
            The updated deployment after it has stopped.

        Raises:
            DeploymentNotRunning:
                If the deployment is not mapped to a :class:`SingleModelDeploymentControl`.
            NotCreatorOwner:
                If the user attempting to stop the deployment is not the creator.
        """
        if not self.is_deployment_running(deployment_id):
            raise DeploymentNotRunning(
                "Deployment must be either running or idle to be stopped"
            )

        if self.deployments[deployment_id].deployment.created_by_id != user_id:
            raise NotCreatorOwner("Only the creator of the deployment can stop it")

        return self.deployments[deployment_id].stop()

    def make_prediction(self, deployment_id: int, x: Any) -> Any:
        """Makes a prediction using the deployment model with the given
        deployment_id and input data x.
        
        A deployment should be in RUNNING state to make predictions.

        Args:
            deployment_id (int): The ID of the deployment to use for prediction.
            x (Any): The input data for the prediction.

        Returns:
            The prediction generated by the deployment model.

        Raises:
            DeploymentNotRunning:
                If the deployment is not mapped to a :class:`SingleModelDeploymentControl`.
        """
        if not self.is_deployment_running(deployment_id):
            raise DeploymentNotRunning("Deployment must be running to make predictions")

        return self.deployments[deployment_id].predict(x)

    def load(self):
        res = requests.post(
            f"{settings.SERVER_HOST}/api/v1/deployments/deployment-manager",
            json=DeploymentManagerComunication(first_init=True).dict(),
            headers={"Authorization": f"Bearer {settings.APPLICATION_SECRET}"},
        )
        deployments = map(lambda x: Deployment(**x), res.json())
        for deployment in deployments:
            try:
                self.add_deployment(deployment)
                self.start_deployment(deployment.id, deployment.created_by_id)
            except Exception as e:
                print(e)
        return True

    @staticmethod
    def get_idle_deployments(deployments: Dict[int, SingleModelDeploymentControl]) -> List[int]:
        return [
            deployment_id
            for deployment_id, deployment in deployments.items()
            if deployment.deployment.status == DeploymentStatus.IDLE
        ]

    def check_idle_long_loop(self):
        while True:
            sleep(5)
            idle_deployments = self.get_idle_deployments(self.deployments)
            for deployment_id in idle_deployments:
                if self.deployments[deployment_id].idle_long:
                    self.remove_deployment(deployment_id)


manager = None
Manager = NewType("Remote Deployment Manager", DeploymentsManager)


def get_deployments_manager() -> Manager:
    """Make sure that only one instance of the manager is created."""
    global manager
    if not manager:
        manager = DeploymentsManager.remote()
    return manager
