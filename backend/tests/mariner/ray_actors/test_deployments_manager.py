from typing import Dict

import pytest
import torch

from mariner.entities.deployment import DeploymentStatus
from mariner.ray_actors.deployments_manager import get_deployments_manager
from mariner.schemas.deployment_schemas import Deployment


@pytest.mark.integration
class TestDeploymentsManager:
    @pytest.mark.asyncio
    async def test_add_deployment(self, some_deployment: Deployment):
        manager = get_deployments_manager()
        deployment: Deployment = await manager.add_deployment.remote(some_deployment)
        assert isinstance(deployment, Deployment)
        assert deployment.status == DeploymentStatus.IDLE
        assert bool(deployment.id in await manager.get_deployments.remote())

    @pytest.mark.asyncio
    async def test_remove_deployment(self, some_deployment: Deployment):
        manager = get_deployments_manager()
        deployment: Deployment = None

        deployment = await manager.add_deployment.remote(some_deployment)
        assert some_deployment.id in await manager.get_deployments.remote()

        await manager.remove_deployment.remote(some_deployment.id)
        assert deployment.id not in await manager.get_deployments.remote()

    @pytest.mark.asyncio
    async def test_start_deployment(self, some_deployment: Deployment):
        manager = get_deployments_manager()
        deployment = await manager.add_deployment.remote(some_deployment)
        deployment = await manager.start_deployment.remote(
            some_deployment.id, some_deployment.created_by_id
        )
        assert deployment.status == DeploymentStatus.ACTIVE
        assert deployment.id in await manager.get_deployments.remote()

    @pytest.mark.asyncio
    async def test_stop_deployment(self, some_deployment: Deployment):
        manager = get_deployments_manager()
        deployment = await manager.add_deployment.remote(some_deployment)
        deployment = await manager.start_deployment.remote(
            some_deployment.id, some_deployment.created_by_id
        )
        assert deployment.status == DeploymentStatus.ACTIVE

        deployment = await manager.stop_deployment.remote(
            some_deployment.id, some_deployment.created_by_id
        )
        assert deployment.status == DeploymentStatus.IDLE

    @pytest.mark.asyncio
    async def test_make_prediction(self, some_deployment: Deployment):
        manager = get_deployments_manager()
        deployment = await manager.add_deployment.remote(some_deployment)
        deployment = await manager.start_deployment.remote(
            some_deployment.id, some_deployment.created_by_id
        )
        assert deployment.status == DeploymentStatus.ACTIVE

        data = {
            "smiles": [
                "CCCC",
                "CCCCC",
                "CCCCCCC",
            ],
            "mwt": [3, 1, 9],
        }
        prediction: Dict[str, torch.Tensor] = await manager.make_prediction.remote(
            some_deployment.id, data
        )
        assert "tpsa" in prediction, "'tpsa' column should be in prediction"
        assert isinstance(
            prediction["tpsa"], torch.Tensor
        ), "'tpsa' column should be a torch.Tensor"
        assert prediction["tpsa"].shape == (3,), "'tpsa' column should have shape (3,)"
