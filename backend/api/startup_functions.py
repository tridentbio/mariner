from fleet.ray_actors.deployments_manager import get_deployments_manager
from mariner.db.session import SessionLocal
from mariner.entities.deployment import Deployment, DeploymentStatus
from mariner.schemas.deployment_schemas import Deployment as DeploymentSchema


async def deployments_manager_startup():
    """Startup function for the :class:`fleet.ray_actors.DeploymentsManager`.

    Should be called when the server starts.

    Deployment Manager initialization flow:
    1. Get all deployments that were running before the server was stopped (status = ACTIVE).
    3. Load all deployments that were running in :class:`fleet.ray_actors.DeploymentsManager`.
    2. Stop all other deployments in the database (status in IDLE, STOPPED, STARTING).
    """
    db = SessionLocal()

    # Map all deployments that were running in the server before it was stopped.
    deployments = DeploymentSchema.from_orm_array(
        db.query(Deployment).filter(Deployment.status == DeploymentStatus.ACTIVE).all()
    )

    # Load all deployments that were running in the new deployments manager instance.
    if len(deployments):
        manager = get_deployments_manager()
        await manager.load_deployments.remote(deployments)

    # Stop all other deployments since they were not running when the server was stopped.
    db.execute(
        f"UPDATE deployment SET status = :stopped where status <> :active",
        {
            "stopped": DeploymentStatus.STOPPED.value.upper(),
            "active": DeploymentStatus.ACTIVE.value.upper(),
        },
    )
    db.close()
