"""
Events service
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from mariner.entities.event import EventEntity, EventSource
from mariner.entities.user import User as UserEntity
from mariner.schemas.api import ApiBaseModel
from mariner.schemas.event_schemas import Event
from mariner.stores.event_sql import EventCreateRepo, event_store


class EventsbySource(ApiBaseModel):
    """Generic event that is produced by interacting with the application.

    Attributes:
        source: specifies kind of event.
        total: number of events of that kind.
        message: description of the grouped events.
        events: detailed events of kind ``source`` that were grouped.
    """

    source: EventSource
    total: int
    message: str
    events: List[Event]


def build_message(source: EventSource, events: List[EventEntity]) -> str:
    """Creates a generic English message to summarize all events

    Args:
        source: the kind of all events
        events: list of events

    Returns:
        English message for the user

    Raises:
        NotImplementedError: when the message is not handled for events of source
    """
    if source == "training:completed":
        assert len(events) > 0, "events argument cannot be empty"
        last_event = events[-1]
        experiment_name = last_event.payload["experiment_name"]
        if len(events) == 2:
            and_others = f' and "{events[0].payload["experiment_name"]}"'
        else:
            and_others = f" and {len(events)-1} others" if len(events) > 1 else ""
        return f'Training "{experiment_name}"{and_others} completed'
    elif source == "changelog":
        return "Checkout what's new!"
    elif "dataset" in source:
        return f"New dataset available. result: {source.split(':')[1]}"
    raise NotImplementedError(f'No message building for source = "{source}"')


def get_events_from_user(db: Session, user: UserEntity):
    """Get unread all events from the user grouped by source"""
    events_grouped = event_store.get_events_by_source(db, user.id)
    payload: List[EventsbySource] = [
        EventsbySource(
            source=key,
            total=len(events),
            message=build_message(key, events),
            events=[Event.from_orm(evt) for evt in events],
        )
        for key, events in events_grouped.items()
    ]
    return payload


def set_events_read(db: Session, user: UserEntity, event_ids: List[int]) -> int:
    """Sets the read flag on the notifications.
    Returns a list of successfully updated events"""
    events = [
        event for event in event_store.get_to_user(db, user.id) if event.id in event_ids
    ]
    updated_count = event_store.update_read(db, events, user.id)
    return updated_count


class EventCreate(ApiBaseModel):
    """A payload to describe a new event"""

    user_id: Optional[int] = None
    timestamp: datetime
    source: EventSource
    payload: Dict[str, Any]
    url: Optional[str] = None


def create_event(db: Session, event: EventCreate):
    """Used by other controllers to register some event"""
    values = event.dict()
    creation_obj = EventCreateRepo.construct(**values)
    newevent = event_store.create(db, obj_in=creation_obj)
    if not event:
        raise RuntimeError("event was not created")
    return Event.from_orm(newevent)
