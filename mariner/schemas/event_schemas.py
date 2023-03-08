"""
Event related DTOs
"""

from typing import Any, Dict, Optional

from mariner.entities import EventSource
from mariner.schemas.api import ApiBaseModel, utc_datetime


class Event(ApiBaseModel):
    """Models an event produced while interacting with the application

    Attributes:
        id: event identifier unique across events
        user_id: id from user that originated this event record
        source: discriminator for event type
        timestamp: time of occurrence
        payload: event details
        url: link for accessing the details page of this event"""

    id: int
    user_id: Optional[int] = None
    source: EventSource
    timestamp: utc_datetime
    payload: Dict[str, Any]
    url: Optional[str] = None
