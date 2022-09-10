from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api import deps
from app.features.events import events_ctl
from app.features.user.model import User
from app.schemas.api import ApiBaseModel

router = APIRouter()


@router.get("/report", response_model=List[events_ctl.EventsbySource])
def get_events_report(
    db: Session = Depends(deps.get_db),
    user: User = Depends(deps.get_current_active_user),
):
    return events_ctl.get_events_from_user(db, user)


class EventsReadResponse(ApiBaseModel):
    total: int


class ReadRequest(ApiBaseModel):
    event_ids: List[int]


@router.post("/read", response_model=EventsReadResponse)
def post_events_read(
    read_request: ReadRequest,
    db: Session = Depends(deps.get_db),
    user: User = Depends(deps.get_current_active_user),
):
    import logging

    logging.error(read_request)
    return EventsReadResponse(
        total=events_ctl.set_events_read(db, user, read_request.event_ids)
    )
