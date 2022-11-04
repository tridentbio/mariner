from fastapi.routing import APIRouter

from api.api_v1.endpoints import (
    data,
    datasets,
    events,
    experiments,
    login,
    model,
    oauth,
    units,
    users,
)

api_router = APIRouter()

api_router.include_router(login.router, tags=["login"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(data.router, prefix="", tags=["utils"])
api_router.include_router(model.router, prefix="/models", tags=["models"])
api_router.include_router(
    experiments.router, prefix="/experiments", tags=["experiments"]
)
api_router.include_router(units.router, prefix="/units", tags=["units"])
api_router.include_router(events.router, prefix="/events", tags=["events"])
api_router.include_router(oauth.router, tags=["oauth"])
