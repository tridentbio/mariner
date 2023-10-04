"""
Assembles the fastapi app instance
"""
import logging

from fastapi.applications import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi_utils.openapi import simplify_operation_ids
from starlette.middleware.cors import CORSMiddleware

from api import startup_functions
from api.api_v1.api import api_router
from api.websocket import ws_router
from fleet.options import options_manager
from mariner.core.config import get_app_settings

app = FastAPI(
    title=get_app_settings("package").name,
    openapi_url="/api/v1/openapi.json",
)


## Filters the healthcheck from logs
logging.getLogger("uvicorn.access").addFilter(
    lambda record: record.args[2] != "/health"
)


@app.on_event("startup")
async def startup_event():
    """
    Runs the startup functions once the server is started.
    """
    options_manager.import_libs()
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


@app.get("/metadata", include_in_schema=False)
def metadata():
    """
    Returns the API metadata: version, tenant, etc.
    """
    package_settings = get_app_settings("package")
    tenant_settings = get_app_settings("tenant")
    return {
        "name": package_settings.name,
        "version": package_settings.version,
        "description": package_settings.description,
        "tenant": {
            "name": tenant_settings.name,
        },
    }


# Set all CORS enabled origins
if get_app_settings("server").cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            str(origin) for origin in get_app_settings("server").cors
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.add_middleware(GZipMiddleware, minimum_size=100)

app.include_router(api_router, prefix="/api/v1")
app.include_router(ws_router)

simplify_operation_ids(app)
