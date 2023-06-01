"""
Package defines all websocket related functionality.
"""
import logging
from typing import Any, Dict, Literal, List

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

    type: Literal[
        "pong", "update-running-metrics", "dataset-process-finish", "update-deployment"
    ]
    data: Any


class UserConnection:
    """Represents a user connection to the server

    Attributes:
        sessions: list of user sessions connected to the server.
    """
    sessions: List[WebSocket]
    def __init__(self, session: WebSocket = None):
        self.sessions = [session] if session else []

    @property
    def is_active(self) -> bool:
        """Returns True if there are active connections"""
        return any([
            session.application_state == WebSocketState.CONNECTED
            for session in self.sessions
        ])

    async def send_message(self, message: WebSocketMessage):
        """Sends a message to all active connections"""
        for session in self.sessions:
            await session.send_text(message.json(by_alias=True))
    
    def add_session(self, session: WebSocket):
        """Adds a session to the active connections"""
        self.sessions.append(session)
        return len(self.sessions) - 1

    def remove_session(self, session_idx: int):
        """Removes a session from the active connections"""
        self.sessions.pop(session_idx)


class ConnectionManager:
    """Manages all active connections to the server
    
    Attributes:
        active_connections (Dict[int, :class:`UserConnection`])): 
            dictionary mapping user ids to their websockets.
    """
    def __init__(self):
        self.active_connections: Dict[int, UserConnection] = {}

    async def connect(self, user_id: int, websocket: WebSocket):
        """Accepts the connection and track the user id associated with it

        Args:
            user_id: id from user on connected
            websocket: instance of the server's socket keeping
            the connection
            
        Returns:
            int: index of the connection in the list of connections
        """
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = UserConnection(websocket)
            return 0
        else:
            return self.active_connections[user_id].add_session(websocket)
        

    def disconnect_session(self, user_id: int, session_idx: int):
        """Removes the connection from an user

        Args:
            user_id: id from user to have websocket disconnected
        """
        if user_id in self.active_connections:
            self.active_connections[user_id].remove_session(session_idx)

            if not self.active_connections[user_id].is_active:
                del self.active_connections[user_id]


    async def send_message(self, user_id: int, message: WebSocketMessage):
        """Sends a message to a specific user

        If the user is not connected, the message is not sent without raising an error

        Args:
            user_id (int): id from user to send the message
            message (WebSocketMessage): message to be sent
        """
        if user_id not in self.active_connections:
            return

        await self.active_connections[user_id].send_message(message)

    async def broadcast(self, message: WebSocketMessage):
        """Sends message to all active connections.

        Args:
            message: message to send.
        """
        for connection in self.active_connections.values():
            await connection.send_message(message)


_manager = None


def get_websockets_manager() -> ConnectionManager:
    """Returns the singleton instance of the connection manager

    Returns:
        ConnectionManager: singleton instance of the connection manager
    """
    global _manager
    if not _manager:
        _manager = ConnectionManager()
    return _manager


ws_router = APIRouter()


@ws_router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket, user: User = Depends(deps.get_current_websocket_user)
):
    """Endpoint for websocket connections

    Args:
        websocket (WebSocket): websocket instance
        user (User, optional):
            user instance.
            Defaults to Depends(deps.get_current_websocket_user).
    """
    manager = get_websockets_manager()
    session_idx = await manager.connect(user.id, websocket)
    while websocket.application_state == WebSocketState.CONNECTED:
        try:
            await websocket.receive_text()  # continuously wait for a message
        except (WebSocketDisconnect, AssertionError):
            break
    manager.disconnect_session(user.id, session_idx)
