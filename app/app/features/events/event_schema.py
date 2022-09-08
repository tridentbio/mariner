from datetime import datetime
from typing import Optional

from app.features.events.event_model import EventSource
from app.schemas.api import ApiBaseModel


class Event(ApiBaseModel):
    id: int
    user_id: Optional[int]
    source: EventSource
    timestamp: datetime
