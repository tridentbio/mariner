import datetime
from typing import List

import pytest
from sqlalchemy.orm import Session

from app.features.events.event_crud import EventCreateRepo, events_repo
from app.features.events.event_model import EventEntity
from app.features.experiments.schema import Experiment
from app.tests.conftest import get_test_user


class TestEventCrud:
    @pytest.fixture(scope="class")
    def events_fixture(self, db: Session, some_experiments: List[Experiment]):
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
        yield events
        db.commit()
        for event in events:
            db.delete(event)
        db.flush()

    def test_get_from_user(self, db: Session, events_fixture: List[EventEntity]):
        user = get_test_user(db)
        assert len(events_repo.get_from_user(db, user)) == len(events_fixture)

    def test_set_events_read(self, db: Session, events_fixture: List[EventEntity]):
        events_slice = events_fixture[:2]
        user = get_test_user(db)
        assert events_repo.update_read(db, events_slice, user.id) == 2
        assert events_repo.update_read(db, events_fixture, user.id) == len(
            events_fixture
        ) - len(events_slice)

    def test_get_event_by_source(self):
        ...
