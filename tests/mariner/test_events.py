from datetime import datetime
from typing import List

from sqlalchemy.orm import Session

from mariner.entities import EventEntity
from mariner import events as events_ctl
from tests.conftest import get_test_user


def test_get_events_from_user(db: Session, events_fixture: List[EventEntity]):
    user = get_test_user(db)
    expected = [
        events_ctl.EventsbySource(
            source="training:completed",
            total=3,
            message=f"",
            events=[],
        )
    ]
    got = events_ctl.get_events_from_user(db, user)
    assert len(expected) == len(got)
    assert got[0].source == expected[0].source
    assert got[0].total == expected[0].total
    assert got[0].message.startswith("Training ") and got[0].message.endswith(
        "and 2 others completed"
    )


def test_set_events_read(events_fixture: List[EventEntity], db: Session):
    user = get_test_user(db)
    to_read_ids = [evt.id for evt in events_fixture[:1]]
    count = events_ctl.set_events_read(db, user, to_read_ids)
    assert count == len(
        to_read_ids
    ), '# of events set as read "{count}" does not match # of unreads frmo input "{len(to_read_ids)}"'
    to_read_ids = [evt.id for evt in events_fixture[:2]]
    count = events_ctl.set_events_read(db, user, to_read_ids)
    assert (
        count == 1
    ), '# of events set as read "{count}" does not match # of unreads frmo input "{1}"'


def test_create_event(db: Session):
    user = get_test_user(db)
    event_payload = {"experiment_name": "tananna", "id": 3}
    payload = events_ctl.EventCreate(
        source="training:completed",
        user_id=user.id,
        timestamp=datetime.now(),
        payload=event_payload,
    )
    event = events_ctl.create_event(db, payload)
    assert event
    assert event.user_id == user.id
    assert event.payload == event_payload
