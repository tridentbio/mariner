from fastapi.routing import APIRouter

from app.api.api_v1.endpoints import (
    data,
    datasets,
    deployment,
    login,
    model,
    users,
    utils,
)

api_router = APIRouter()

api_router.include_router(login.router, tags=["login"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(data.router, prefix="", tags=["utils"])
api_router.include_router(model.router, prefix="/models", tags=["models"])
api_router.include_router(
    deployment.router, prefix="/deployments", tags=["model-deployments"]
)


# Legacy from template
api_router.include_router(utils.router, prefix="/utils", tags=["utils"])
