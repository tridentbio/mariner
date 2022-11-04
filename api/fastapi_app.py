from fastapi.applications import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi_utils.openapi import simplify_operation_ids
from starlette.middleware.cors import CORSMiddleware

from api.api_v1.api import api_router
from api.websocket import ws_router
from mariner.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)


@app.get("/health", response_model=str)
def healthcheck():
    return ""


@app.get("/openapi.json", include_in_schema=False)
def openapijson():
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

app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(ws_router)

simplify_operation_ids(app)
