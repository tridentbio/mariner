import uvicorn
from fastapi.applications import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.api.api_v1.api import api_router
from app.api.websocket import ws_router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)


@app.get("/health", response_model=str)
def healthcheck():
    return ""


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

if __name__ == "__main__":
    # same as running  `uvicorn app.main:app --reload --host 0.0.0.0 --port 80` in CLI
    uvicorn.run(app, host="0.0.0.0", port=80)
