import datetime
from typing import List

from sqlalchemy.orm import Session

from app.features.events.event_crud import EventCreateRepo, events_repo
from app.features.events.event_model import EventEntity
from app.features.experiments.schema import Experiment
from app.tests.conftest import get_test_user


def get_test_events(
    db: Session, some_experiments: List[Experiment]
) -> List[EventEntity]:
    user = get_test_user(db)
    mocked_events = [
        EventCreateRepo(
            user_id=user.id,
            source="training:completed",
            timestamp=datetime.datetime(2022, 10, 29),
            payload=experiment.dict(),
        )
        for experiment in some_experiments
    ]
    events = [events_repo.create(db, obj_in=mock) for mock in mocked_events]
    return events


def teardown_events(db: Session, events: List[EventEntity]):
    db.commit()
    for event in events:
        db.delete(event)
    db.flush()
