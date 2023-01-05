from typing import Literal, get_args

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql.functions import current_timestamp

from mariner.db.base_class import Base

EventSource = Literal["training:completed", "changelog"]


class EventReadEntity(Base):
    """Represents the user marked an event as read"""

    event_id = Column(
        Integer, ForeignKey("event.id", ondelete="CASCADE"), primary_key=True
    )
    user_id = Column(Integer, ForeignKey("user.id"), primary_key=True)
    created_at = Column(DateTime, server_default=current_timestamp())
    event = relationship("EventEntity")


class EventEntity(Base):
    """Represents the occurrence of an event

    Attributes:
        id: unique identifier of an event
        user_id(Column[int]): column represent the user that originated the event
            A null user_id means it's a global event
        timestamp(Column[datetime.datetime]): time of the event
        source(Column[str]): Property to distinguish event type
    """

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=True)
    timestamp = Column(DateTime, nullable=False)
    source = Column(String, nullable=False)
    payload = Column(JSON, nullable=True)
    url = Column(String, nullable=True)

    @validates("source")
    def validate_source(self, key, source):
        valid_event_sources = get_args(EventSource)
        if source not in valid_event_sources:
            raise ValueError(f"Invalid source for EventEntity {source}")
        return source
