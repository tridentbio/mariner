import pytest
from fastapi import WebSocket

from api.websocket import (
    AnonymousConnection,
    BaseConnection,
    ConnectionManager,
    UserConnection,
    get_websockets_manager,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="function")
def websocket_fixture():
    async def awaitable(*_):
        return {"type": "websocket.connect"}

    return WebSocket(scope={"type": "websocket"}, receive=awaitable, send=awaitable)


async def test_base_connection(websocket_fixture: WebSocket):
    base_connection = BaseConnection()

    # Test the active state of a new connection
    assert not base_connection.is_active

    session_id = base_connection.add_session(websocket_fixture)
    assert base_connection.sessions == {session_id: websocket_fixture}

    # Remove the WebSocket and check if it's removed correctly
    base_connection.remove_session(session_id)
    assert base_connection.sessions == {}


async def test_user_connection():
    user_connection = UserConnection()
    assert isinstance(user_connection, BaseConnection)


async def test_anonymous_connection():
    anonymous_connection = AnonymousConnection()
    assert isinstance(anonymous_connection, BaseConnection)


async def test_connection_manager_private(websocket_fixture: WebSocket):
    manager = ConnectionManager()
    assert manager.active_connections == {}
    assert manager.active_connections_public == {}

    session_id = await manager.connect(1, websocket_fixture)
    assert 1 in manager.active_connections
    assert manager.active_connections[1].sessions == {session_id: websocket_fixture}

    manager.disconnect_session(1, session_id)
    assert manager.active_connections == {}


async def test_connection_manager_public(websocket_fixture: WebSocket):
    manager = ConnectionManager()
    assert manager.active_connections == {}
    assert manager.active_connections_public == {}

    session_id = await manager.connect(1, websocket_fixture, public=True)
    assert 1 in manager.active_connections_public
    assert manager.active_connections_public[1].sessions == {
        session_id: websocket_fixture
    }

    manager.disconnect_session(1, session_id, public=True)
    assert manager.active_connections_public == {}


def test_get_websockets_manager():
    manager = get_websockets_manager()
    assert isinstance(manager, ConnectionManager)
