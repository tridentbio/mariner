from typing import Any, Dict, Optional

from app.features.events.event_model import EventSource
from app.schemas.api import ApiBaseModel, utc_datetime


class Event(ApiBaseModel):
    id: int
    user_id: Optional[int] = None
    source: EventSource
    timestamp: utc_datetime
    payload: Dict[str, Any]
    url: Optional[str] = None
