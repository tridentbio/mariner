"""
Package defines all websocket related funcionality
"""
import logging
from typing import Any, Dict, Literal

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.param_functions import Depends
from fastapi.routing import APIRouter
from starlette.websockets import WebSocketState

from api import deps
from mariner.entities import User
from mariner.schemas.api import ApiBaseModel

LOG = logging.getLogger(__name__)


class WebSocketMessage(ApiBaseModel):
    """
    Base class for messages exchanged with client
    """

    type: Literal["pong", "update-running-metrics"]
    data: Any


def _is_closed(ws: WebSocket):
    return ws.application_state == WebSocketState.DISCONNECTED


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, WebSocket] = {}

    async def connect(self, user_id: int, websocket: WebSocket):
        """ "Accepts the connection and track the user id associated with it


        Args:
            user_id: id from user on connected
            websocket: instance of the server's socket keeping
            the connection
        """
        if user_id in self.active_connections:
            existing_ws = self.active_connections[user_id]
            if existing_ws and not _is_closed(existing_ws):
                try:
                    await existing_ws.close()
                except Exception:
                    # Error raised before: "Trying to send a websocket.close'
                    # after sending a 'websocket.close'"
                    LOG.error("Error while closing socket.")
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect_session(self, user_id: int):
        """Removes the connection from an user

        Args:
            user_id: id from user to have websocket disconnected
        """
        self.active_connections.pop(user_id)

    async def send_message(self, user_id: int, message: WebSocketMessage):
        print("[ConnectionManager] active connections", user_id, message)
        if user_id not in self.active_connections:
            print("NO USER FOUND")
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
    while websocket.application_state == WebSocketState.CONNECTED:
        try:
            await websocket.receive_text()  # continuously wait for a message
        except (WebSocketDisconnect, AssertionError):
            break
