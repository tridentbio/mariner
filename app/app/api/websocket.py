from typing import Any, Dict, Literal

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.param_functions import Depends
from fastapi.routing import APIRouter
from starlette.websockets import WebSocketState

from app.api import deps
from app.features.user.model import User
from app.schemas.api import ApiBaseModel


class WebSocketMessage(ApiBaseModel):
    type: Literal["pong", "update-running-metrics"]
    data: Any


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, WebSocket] = {}

    async def connect(self, user_id: int, websocket: WebSocket):
        if user_id in self.active_connections:
            ws = self.active_connections[user_id]
            if ws.application_state == WebSocketState.CONNECTED:
                return
            self.active_connections.pop(user_id)
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect_session(self, user_id: int):
        self.active_connections.pop(user_id)

    async def send_message(self, user_id: int, message: WebSocketMessage):
        if user_id not in self.active_connections:
            return
        await self.active_connections[user_id].send_text(message.json())

    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)


_manager = None


def get_websockets_manager() -> ConnectionManager:
    global _manager
    if not _manager:
        _manager = ConnectionManager()
    return _manager


ws_router = APIRouter()


@ws_router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket, user: User = Depends(deps.get_current_websocket_user)
):
    manager = get_websockets_manager()
    await manager.connect(user.id, websocket)
    while True:
        try:
            await websocket.receive_text()  # continuously wait for a message
        except WebSocketDisconnect:
            break
