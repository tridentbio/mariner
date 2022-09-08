from datetime import datetime
from typing import List, Literal, Optional

from sqlalchemy.orm import Session

from app.features.events.event_crud import EventCreateRepo, events_repo
from app.features.events.event_model import EventEntity, EventSource
from app.features.events.event_schema import Event
from app.features.user.model import User as UserEntity
from app.schemas.api import ApiBaseModel


class EventsbySource(ApiBaseModel):
    source: EventSource
    total: int
    message: str


def build_message(source: EventSource, events: List[EventEntity]) -> str:
    if source == "training:completed":
        last_event = events[-1]
        experiment_name = dict(last_event.payload)["experiment_name"]
        and_others = " and other {len(events)-1}" if len(events) > 1 else ""
        return f'Training "{experiment_name}"{and_others} completed'
    raise NotImplemented(f'No message building for source = "{source}"')


def get_events_from_user(db: Session, user: UserEntity):
    """Get unread all events from the user grouped by source"""
    events_grouped = events_repo.get_events_by_source(db, user.id)
    payload: List[EventsbySource] = [
        EventsbySource(
            source=key, total=len(events), message=build_message(key, events)
        )
        for key, events in events_grouped.items()
    ]
    return payload


def set_events_read(db: Session, user: UserEntity, event_ids: List[int]) -> List[int]:
    """Sets the read flag on the notifications.
    Returns a list of succesfully updated events"""
    events = [
        event for event in events_repo.get_from_user(db, user) if event.id in event_ids
    ]
    updated_count = events_repo.update_read(db, events, user.id)
    return updated_count


class EventCreate(ApiBaseModel):
    """A payload to describe a new event"""

    user_id: Optional[int]
    timestamp: datetime
    source: Literal["training:completed"]


def create_event(db: Session, event: EventCreate):
    """Used by other controllers to register some event"""
    values = event.dict()
    event = events_repo.create(db, EventCreateRepo.construct(**values))
    if not event:
        raise RuntimeError("event was not created")
    return Event.from_orm(event)
