from fastapi.routing import APIRouter

from app.api.api_v1.endpoints import datasets, login, users, utils, data

api_router = APIRouter()

api_router.include_router(login.router, tags=["login"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(data.router, prefix="", tags=["utils"])

# Legacy from template
api_router.include_router(utils.router, prefix="/utils", tags=["utils"])

