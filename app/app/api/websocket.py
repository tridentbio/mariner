from typing import Any, Dict, Literal

from fastapi import WebSocket
from fastapi.param_functions import Depends
from fastapi.routing import APIRouter
from starlette.websockets import WebSocketState

from app.api import deps
from app.schemas.api import ApiBaseModel


class WebSocketMessage(ApiBaseModel):
    type: Literal["pong", "update-running-metrics"]
    data: Any


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        if session_id in self.active_connections:
            ws = self.active_connections[session_id]
            if ws.application_state == WebSocketState.CONNECTED:
                await ws.close()
            self.active_connections.pop(session_id)
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect_session(self, session_id: str):
        self.active_connections.pop(session_id)

    async def send_message(self, session_id: str, message: WebSocketMessage):
        await self.active_connections[session_id].send_text(message.json())

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
    websocket: WebSocket, user: str = Depends(deps.get_cookie_or_token)
):
    manager = get_websockets_manager()
    await manager.connect(user, websocket)
    pong_message = WebSocketMessage(type="pong", data=[])
    while True:
        await websocket.receive_text()
        await manager.send_message(user, pong_message)
