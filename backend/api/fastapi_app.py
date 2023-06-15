"""
Assembles the fastapi app instance
"""
from fastapi.applications import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi_utils.openapi import simplify_operation_ids
from starlette.middleware.cors import CORSMiddleware

from api import startup_functions
from api.api_v1.api import api_router
from api.websocket import ws_router
from api.deps_flow import deps_flow_router
from mariner.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)


@app.on_event("startup")
async def startup_event():
    """
    Runs the startup functions once the server is started.
    """
    await startup_functions.deployments_manager_startup()


@app.get("/health", response_model=str)
def healthcheck():
    """
    Server health check
    """
    return ""


@app.get("/openapi.json", include_in_schema=False)
def openapijson():
    """
    Returns our openapi's json
    """
    return get_openapi(title=app.title, version=app.version, routes=app.routes)


# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.add_middleware(GZipMiddleware, minimum_size=100)

app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(ws_router)

if settings.ENV == "development":
    app.include_router(deps_flow_router)

simplify_operation_ids(app)
