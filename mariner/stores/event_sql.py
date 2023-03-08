"""
Event data layer defining ways to read and write to the events collection
"""
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel
from sqlalchemy import and_, or_
from sqlalchemy.orm import Query, Session

from mariner.entities.event import EventEntity, EventReadEntity, EventSource
from mariner.stores.base_sql import CRUDBase


class EventCreateRepo(BaseModel):
    """Payload for inserting an event to the events collection"""

    source: Literal["training:completed"]
    user_id: Optional[int]
    timestamp: datetime
    payload: Any
    url: Optional[str]


class EventCRUD(CRUDBase[EventEntity, EventCreateRepo, None]):
    """Data layer operators on the events collection"""

    def _is_event_to_user(self, query: Query, user_id: int):
        return query.filter(
            or_(EventEntity.user_id.is_(None), EventEntity.user_id == user_id)
        )

    def _with_reads_join(self, q: Query):
        return q.join(
            EventReadEntity, EventReadEntity.event_id == EventEntity.id, isouter=True
        )

    def _filter_unread(self, q: Query):
        return q.filter(EventReadEntity.event_id.is_(None))

    def get_events_by_source(
        self, db: Session, user_id: int
    ) -> Dict[EventSource, List[EventEntity]]:
        """
        Gets the unread events of a user grouped by source
        """

        query = self._with_reads_join(db.query(EventEntity))
        query = self._is_event_to_user(query, user_id=user_id)
        query = self._filter_unread(query)
        events: List[EventEntity] = query.all()
        grouped = {}
        for event in events:
            if event.source not in grouped:
                grouped[event.source] = [event]
            else:
                grouped[event.source].append(event)
        return grouped

    def get_to_user(self, db: Session, user_id: int) -> List[EventEntity]:
        """Gets global (where user_id is null) and personal events of a user"""
        query = db.query(EventEntity)
        query = self._is_event_to_user(query, user_id)
        return query.all()

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

    def get(self, db: Session, from_source: Optional[EventSource] = None):
        query = db.query(EventEntity)
        if from_source:
            query = query.filter(EventEntity.source == from_source)
        return query.all()


event_store = EventCRUD(EventEntity)
