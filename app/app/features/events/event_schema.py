from datetime import datetime
from typing import Any, Dict, Optional

from app.features.events.event_model import EventSource
from app.schemas.api import ApiBaseModel


class Event(ApiBaseModel):
    id: int
    user_id: Optional[int] = None
    source: EventSource
    timestamp: datetime
    payload: Dict[str, Any]
    url: Optional[str] = None
