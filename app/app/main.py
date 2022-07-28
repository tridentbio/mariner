from time import sleep

from fastapi import Depends, WebSocket, WebSocketDisconnect
from fastapi.applications import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.api import deps
from app.api.api_v1.api import api_router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/health", response_model=str)
def healthcheck():
    return ""


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket, cookie_or_token: str = Depends(deps.get_cookie_or_token)
):
    try:
        await websocket.accept()
        await websocket.send_text(
            f"Session cookie or query token value is: {cookie_or_token}"
        )
        while True:
            await websocket.send_text("Query parameter q is")
            sleep(5)
    except WebSocketDisconnect:
        pass


app.include_router(api_router, prefix=settings.API_V1_STR)
