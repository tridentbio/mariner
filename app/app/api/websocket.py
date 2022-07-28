from abc import ABC
from typing import Dict, Optional

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.param_functions import Depends
from fastapi.routing import APIRouter

from app.api import deps
from app.features.user.model import User
from app.schemas.api import ApiBaseModel


class WebSocketMessage(ApiBaseModel, ABC):
    type: str
    data: ApiBaseModel


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect_session(self, session_id: str):
        self.active_connections.pop(session_id)

    async def send_message(self, session_id: str, message: WebSocketMessage):
        await self.active_connections[session_id].send_json(message.json())

    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)


_manager = None


def get_manager() -> ConnectionManager:
    global _manager
    if not _manager:
        _manager = ConnectionManager()
    return _manager


ws_router = APIRouter()


class PongWSData(ApiBaseModel):
    message = "PONG"


class PongWSMessage(WebSocketMessage):
    type = "pong"
    data: Optional[PongWSData]

    def __init__(self):
        self.type = "pong"
        self.data = PongWSData()


@ws_router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket, user: User = Depends(deps.get_current_active_user)
):
    print("hello from /ws")
    manager = get_manager()
    await manager.connect(user.email, websocket)
    try:
        while True:
            msg = await websocket.receive_json()
            print(f"User {user.email} send {repr(msg)}")
            pong_message = PongWSMessage()
            await manager.send_message(user.email, pong_message)
    except WebSocketDisconnect:
        manager.disconnect_session(user.email)
