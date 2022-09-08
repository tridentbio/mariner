from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.features.events.event_model import (
    EventEntity,
    EventReadEntity,
    EventSource,
)
from app.features.user.model import User as UserEntity


class EventCreateRepo(BaseModel):
    source: Literal["training:completed"]
    user_id: Optional[int]
    timestamp: datetime
    payload: Any


class EventUpdateRepo(BaseModel):
    pass


class EventCRUD(CRUDBase[EventEntity, EventCreateRepo, EventUpdateRepo]):
    def get_events_by_source(
        self, db: Session, user_id: int
    ) -> Dict[EventSource, List[EventEntity]]:
        """
        Gets the events of a user grouped by source
        """
        return {}

    def get_from_user(self, db: Session, user: UserEntity) -> List[EventEntity]:
        return (
            db.query(EventEntity)
            .filter(or_(EventEntity.user_id.is_(None), EventEntity.user_id == user.id))
            .all()
        )

    def update_read(self, db: Session, dbobjs: List[EventEntity], user_id: int) -> int:
        event_ids = [dbobj.id for dbobj in dbobjs]
        reads = (
            db.query(EventReadEntity)
            .filter(
                and_(
                    EventReadEntity.event_id.in_(event_ids),
                    EventReadEntity.user_id == user_id,
                )
            )
            .all()
        )
        read_event_ids = [read.event_id for read in reads]
        updated = 0
        for dbobj in dbobjs:
            if dbobj.id not in read_event_ids:
                db.add(EventReadEntity(event_id=dbobj.id, user_id=user_id))
                updated += 1
        db.flush()
        db.commit()
        return updated


events_repo = EventCRUD(EventEntity)
