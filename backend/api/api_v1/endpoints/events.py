"""
Handlers for api/v1/events* endpoints
"""
from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

import mariner.events as events_ctl
from api import deps
from mariner.entities.user import User
from mariner.schemas.api import ApiBaseModel

router = APIRouter()


@router.get("/report", response_model=List[events_ctl.EventsbySource])
def get_events_report(
    db: Session = Depends(deps.get_db),
    user: User = Depends(deps.get_current_active_user),
):
    """Endpoint to get user unread notifications.

    Args:
        db: Connection to the database.
        user: User that originated the request.

    Returns:
        Returns a list of unread notifications grouped by notification
        kind.
    """
    return events_ctl.get_events_from_user(db, user)


class EventsReadResponse(ApiBaseModel):
    """Models response of a command that marks events as read.

    Attributes:
        total: number of events updated.
    """

    total: int


class ReadRequest(ApiBaseModel):
    """Models a request to mark events as read.

    Attributes:
        event_ids: ids of events to mark as read."""

    event_ids: List[int]


@router.post("/read", response_model=EventsReadResponse)
def post_events_read(
    read_request: ReadRequest,
    db: Session = Depends(deps.get_db),
    user: User = Depends(deps.get_current_active_user),
):
    """Endpoint to mark events (notification) as read.

    Args:
        read_request: payload specifying events to be updated.
        db: Connection to the database.
        user: User that originated request.

    Returns:
        Count of updated events.
    """
    return EventsReadResponse(
        total=events_ctl.set_events_read(db, user, read_request.event_ids)
    )
