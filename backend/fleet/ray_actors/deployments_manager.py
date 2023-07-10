from threading import Thread
from time import sleep, time
from typing import Any, Dict, List, Optional, Union

import ray
import requests

from fleet.mlflow import load_pipeline
from fleet.scikit_.model_functions import SciKitFunctions
from fleet.torch_.model_functions import TorchFunctions
from mariner.core import mlflowapi
from mariner.core.config import get_app_settings
from mariner.entities.deployment import DeploymentStatus
from mariner.exceptions import (
    DeploymentNotRunning,
    ModelVersionNotFound,
    ModelVersionNotTrained,
    NotCreatorOwner,
)
from mariner.schemas.deployment_schemas import (
    Deployment,
    DeploymentManagerComunication,
)


class SingleModelDeploymentControl:
    """Responsible for managing the deployment of a model.

    Used by the :class:`DeploymentsManager` to manage the lifecycle of a deployment.

    Possible status of a deployment:

    - IDLE: The deployment already have a :class:`SingleModelDeploymentControl` but it is not running.
    - STARTING: The deployment is starting.
    - ACTIVE: The deployment was loaded from mlflow and it's running.
    - STOPPED: The deployment don't have a :class:`SingleModelDeploymentControl`.

    Attributes:
        deployment (Deployment): The deployment to be managed.
        is_running (bool): Real status of deployment in the manager.
            More trustworthy than the deployment status, since it is updated
            by a third party.
        model (Optional[CustomModel]): The model loaded from mlflow.
        idle_time (Optional[int]): The time when the deployment became idle.
    """

    deployment: Deployment
    is_running: bool
    functions: Union[SciKitFunctions, TorchFunctions, None] = None
    mlflow_modelversion = None
    idle_time: Optional[int] = None

    def __init__(self, deployment: Deployment, quiet: bool = False):
        """Initialization method.

        Args:
            deployment (Deployment): The deployment to be managed.
            quiet (bool, optional):
                Defines if the instance should update the deployment status in the database on init.
                True if it should not, False otherwise.
                Useful when the instance is being loaded from the database and the server is
                not running yet.
        """
        self.deployment = deployment
        if not quiet:
            self.update_status(DeploymentStatus.IDLE)

        self.mlflow_modelversion = mlflowapi.get_model_version(
            model_name=deployment.model_version.mlflow_model_name,
            version=deployment.model_version.mlflow_version,
        )

        self.is_running = False
        self.functions = None

    @property
    def idle_long(self) -> bool:
        """Checks if the deployment is idle for a long time.

        Returns:
            True if the deployment is idle for a long time, False otherwise.
        """
        return (
            self.idle_time
            and (time() - self.idle_time)
            > get_app_settings().DEPLOYMENT_IDLE_TIME
        )

    def load_functions(self):
        """Loads the functions of the model based on its framework."""
        if self.deployment.model_version.config.framework == "torch":
            model = mlflowapi.get_model_by_uri(
                self.deployment.model_version.get_mlflow_uri()
            )
            self.functions = TorchFunctions(
                spec=self.deployment.model_version.config, model=model
            )

        elif self.deployment.model_version.config.framework == "scikit":
            model = mlflowapi.get_model_by_uri(
                self.deployment.model_version.get_mlflow_uri()
            )
            pipeline = load_pipeline(self.mlflow_modelversion.run_id)
            self.functions = SciKitFunctions(
                spec=self.deployment.model_version.config,
                model=model,
                preprocessing_pipeline=pipeline,
            )

        self.is_running = True

    def start(self) -> Deployment:
        """Starts the SingleModelDeploymentControl, which includes loading the
        model and updating its status to ACTIVE.

        Returns:
            The updated deployment after starting the instance.
        """
        self.update_status(DeploymentStatus.STARTING)
        self.load_functions()
        self.update_status(DeploymentStatus.ACTIVE)
        return self.deployment

    def stop(self) -> Deployment:
        """Stops the SingleModelDeploymentControl by setting ending its model
        cycle, updating its running status to False, and setting its
        status to IDLE.

        Returns:
            The updated deployment after stopping the instance.
        """
        self.functions = None
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

        return self.functions.predict(x)

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
            f"{get_app_settings().SERVER_HOST}/api/v1/deployments/deployment-manager",
            json=data.dict(),
            headers={
                "Authorization": f"Bearer {get_app_settings().APPLICATION_SECRET}"
            },
        )
        assert (
            res.status_code == 200,
            f"Request to update deployment failed with status  {res.status_code}",
        )

        self.deployment = Deployment(**res.json())

    @classmethod
    def build_and_run(cls, deployment):
        """Method used by the :class:`DeploymentsManager` to build a new instance
        of :class:`SingleModelDeploymentControl` and start it.

        Since it will not use the database to store the deployment, this method should
        be used on first initialization of the deployment manager for all deployments
        that already have status ACTIVE in the database.

        Args:
            deployment (Deployment): The deployment to be managed.
        """
        instance = cls(deployment, quiet=True)
        instance.load_functions()
        return instance


@ray.remote
class DeploymentsManager:
    """Ray Actor responsible for managing multiple instances
    of :class:`SingleModelDeploymentControl` and handling their life cycle
    (starting, stopping, etc.). Should be used as a singleton.

    .. warning:: Should not be instantiated directly, call :func:`get_deployments_manager` instead.
    """

    deployments_map: Dict[int, SingleModelDeploymentControl]

    def __init__(self):
        self.deployments_map = {}
        Thread(target=self.check_idle_long_loop).start()

    def get_deployments(self):
        """Returns a dictionary containing all the registered
        :class:`SingleModelDeploymentControl` objects.
        """
        return self.deployments_map

    def is_deployment_running(self, deployment_id: int) -> bool:
        """Checks if the specified deployment is running on
        current ray actor.

        Args:
            deployment_id (int): The ID of the deployment to check.

        Returns:
            True if the deployment is running, False otherwise.
        """
        return (
            deployment_id in self.deployments_map.keys()
            and self.deployments_map[deployment_id].is_running
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

        if not deployment.id in self.deployments_map.keys():
            self.deployments_map[deployment.id] = SingleModelDeploymentControl(
                deployment
            )
            assert (
                self.deployments_map[deployment.id].deployment.status
                == DeploymentStatus.IDLE,
                "Deployment must be idle when added.",
            )
        return self.deployments_map[deployment.id].deployment

    def remove_deployment(self, deployment_id: int):
        """Removes a deployment from the DeploymentsManager.

        A deployment should be in STOPPED state after removed.

        Args:
            deployment_id (int): The ID of the deployment to be removed.

        Raises:
            DeploymentNotRunning:
                If the deployment is not mapped to a :class:`SingleModelDeploymentControl`.
        """
        if not deployment_id in self.deployments_map.keys():
            raise DeploymentNotRunning(
                "Deployment must be either running or idle to be removed"
            )
        self.deployments_map[deployment_id].update_status(
            DeploymentStatus.STOPPED
        )
        del self.deployments_map[deployment_id]

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
        if not deployment_id in self.deployments_map.keys():
            raise DeploymentNotRunning(
                "Deployment must be either stopped or idle to be started"
            )

        if self.deployments_map[deployment_id].is_running:
            return self.deployments_map[deployment_id].deployment

        if (
            self.deployments_map[deployment_id].deployment.created_by_id
            != user_id
        ):
            raise NotCreatorOwner(
                "Only the creator of the deployment can start it"
            )

        return self.deployments_map[deployment_id].start()

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

        if (
            self.deployments_map[deployment_id].deployment.created_by_id
            != user_id
        ):
            raise NotCreatorOwner(
                "Only the creator of the deployment can stop it"
            )

        return self.deployments_map[deployment_id].stop()

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
            raise DeploymentNotRunning(
                "Deployment must be running to make predictions"
            )

        return self.deployments_map[deployment_id].predict(x)

    def load_deployments(self, deployments: List[Deployment]):
        """Used to start a list of deployments when the DeploymentsManager.

        Should be used on first initialization of the DeploymentsManager to load
        all deployments from the database that were running before the server was
        stopped.

        Args:
            deployments (List[Deployment]): The list of deployments to start.
        """
        for deployment in deployments:
            if deployment.id not in self.deployments_map.keys():
                self.deployments_map[
                    deployment.id
                ] = SingleModelDeploymentControl.build_and_run(deployment)

    @staticmethod
    def get_idle_deployments(
        deployments: Dict[int, SingleModelDeploymentControl]
    ) -> List[int]:
        return [
            deployment_id
            for deployment_id, deployment in deployments.items()
            if deployment.deployment.status == DeploymentStatus.IDLE
        ]

    def check_idle_long_loop(self):
        while True:
            sleep(5)
            idle_deployments = self.get_idle_deployments(self.deployments_map)
            for deployment_id in idle_deployments:
                if self.deployments_map[deployment_id].idle_long:
                    self.remove_deployment(deployment_id)


manager = None


def get_deployments_manager() -> DeploymentsManager:
    """Make sure that only one instance of the manager is created."""
    global manager
    if not manager:
        manager = DeploymentsManager.remote()
    return manager
