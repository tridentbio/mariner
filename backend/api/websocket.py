"""
Package defines all websocket related functionality.
"""
import logging
from typing import Any, Dict, List, Literal
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.param_functions import Depends
from fastapi.routing import APIRouter
from sqlalchemy.orm import Session
from starlette.websockets import WebSocketState

from api import deps
from mariner.entities import Deployment, User
from mariner.schemas.api import ApiBaseModel

LOG = logging.getLogger(__name__)


class WebSocketResponse(ApiBaseModel):
    """
    Base class for messages exchanged with client
    """

    type: Literal[
        "pong",
        "update-running-metrics",
        "dataset-process-finish",
        "update-deployment",
        "update-model",
    ]
    data: Any


class BaseConnection:
    """Represents a base connection to the server.

    Each connection is identified by a session_id.

    Attributes:
        sessions: sessions connected to the server.
    """

    sessions: Dict[str, WebSocket]

    def __init__(self):
        self.sessions = {}

    @property
    def is_active(self) -> bool:
        """Returns True if there are active connections."""
        return any(
            session.application_state == WebSocketState.CONNECTED
            for session in self.sessions.values()
        )

    async def send_message(self, message: WebSocketResponse):
        """Sends a message to all active connections."""
        for session in self.sessions.values():
            if session.application_state == WebSocketState.CONNECTED:
                await session.send_text(message.json(by_alias=True))

    def add_session(self, session: WebSocket):
        """Adds a session to the active connections."""
        session_id = uuid4().hex
        self.sessions[session_id] = session
        return session_id

    def remove_session(self, session_id: str):
        """Removes a session from the active connections."""
        self.sessions.pop(session_id)


class UserConnection(BaseConnection):
    """Represents a user connected to the server."""


class AnonymousConnection(BaseConnection):
    """Represents a public deployment with some anonymous
    users connected to the server.
    """


class ConnectionManager:
    """Manages all active connections to the server

    Attributes:
        active_connections (Dict[int, UserConnection])):
            dictionary mapping user ids to their websockets.
    """

    def __init__(self):
        self.active_connections: Dict[int, UserConnection] = {}
        self.active_connections_public: Dict[int, AnonymousConnection] = {}

    async def connect(
        self, connection_id: int, websocket: WebSocket, public: bool = False
    ):
        """Accepts the connection and track the user id associated with it.

        Args:
            connection_id:
                for authenticated users: id of the user.
                for anonymous users in public deployments: id of the deployment.
            websocket: instance of the server's socket keeping the connection.
            public: whether the connection is public or not.

        Returns:
            str: id of the connection in the list of connections.
        """
        await websocket.accept()

        if public:
            if connection_id not in self.active_connections_public:
                self.active_connections_public[
                    connection_id
                ] = AnonymousConnection()

            return self.active_connections_public[connection_id].add_session(
                websocket
            )

        else:
            if connection_id not in self.active_connections:
                self.active_connections[connection_id] = UserConnection()

            return self.active_connections[connection_id].add_session(
                websocket
            )

    def disconnect_session(
        self, connection_id: int, session_id: str, public: bool = False
    ):
        """Removes the connection from an user

        Args:
            connection_id: id from connection to have websocket disconnected
            session_id: id from session to be disconnected
            public: whether the connection is public or not.
        """
        if public:
            if connection_id in self.active_connections_public:
                self.active_connections_public[connection_id].remove_session(
                    session_id
                )

                if not self.active_connections_public[connection_id].is_active:
                    self.active_connections_public.pop(connection_id)

        else:
            if connection_id in self.active_connections:
                self.active_connections[connection_id].remove_session(
                    session_id
                )

                if not self.active_connections[connection_id].is_active:
                    self.active_connections.pop(connection_id)

    async def send_message_to_user(
        self, user_id: int | List[int], message: WebSocketResponse
    ):
        """Sends a message to a specific user

        If the user is not connected, the message is not sent without raising an error

        Args:
            user_id (int): id from user to send the message
            message (WebSocketMessage): message to be sent
        """
        if isinstance(user_id, int):
            user_id = [user_id]
        for id_ in user_id:
            if id_ not in self.active_connections:
                return

            await self.active_connections[id_].send_message(message)

    async def broadcast(
        self, message: WebSocketResponse, public: bool = False
    ):
        """Sends message to all active connections.

        Args:
            message: message to send.
            public: If True, sends the message to all connections, send only to private connections otherwise.
        """
        for connection in [
            *self.active_connections.values(),
            *(self.active_connections_public.values() if public else []),
        ]:
            await connection.send_message(message)


_manager = None


def get_websockets_manager() -> ConnectionManager:
    """Returns the singleton instance of the connection manager

    Returns:
        ConnectionManager: singleton instance of the connection manager
    """
    global _manager  # pylint: disable=global-statement
    if not _manager:
        _manager = ConnectionManager()
    return _manager


ws_router = APIRouter()


@ws_router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    db: Session = Depends(deps.get_db),
    token: str = Depends(deps.get_cookie_or_token),
):
    """Endpoint for websocket connections

    Args:
        websocket (WebSocket): websocket instance
        user (User, optional):
            user instance.
            Defaults to Depends(deps.get_current_websocket_user).
    """
    try:
        user = deps.get_current_user(db, token)
    except Exception as exc:
        await websocket.close()
        return

    manager = get_websockets_manager()
    session_id = await manager.connect(user.id, websocket)
    while websocket.application_state == WebSocketState.CONNECTED:
        try:
            text = (
                await websocket.receive_text()
            )  # continuously wait for a message
            LOG.warning("[WEBSOCKET]: %s", text)

        except (WebSocketDisconnect, AssertionError):
            break

    manager.disconnect_session(user.id, session_id)


@ws_router.websocket("/ws-public")
async def public_websocket_endpoint(
    websocket: WebSocket,
    deployment: Deployment = Depends(deps.get_current_websocket_deployment),
):
    """Endpoint for websocket connections

    Restricted to public deployments.

    Args:
        websocket: websocket instance
        deployment: deployment entity of public deployment that cames from the token.
    """
    manager = get_websockets_manager()
    session_id = await manager.connect(deployment.id, websocket, public=True)
    while websocket.application_state == WebSocketState.CONNECTED:
        try:
            await websocket.receive_text()  # continuously wait for a message
        except (WebSocketDisconnect, AssertionError):
            break
    manager.disconnect_session(deployment.id, session_id, public=True)
