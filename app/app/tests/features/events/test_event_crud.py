from typing import List

import pytest
from sqlalchemy.orm import Session

from app.features.events.event_crud import events_repo
from app.features.events.event_model import EventEntity
from app.features.events.event_schema import Event
from app.features.experiments.schema import Experiment
from app.tests.conftest import get_test_user
from app.tests.features.events.utils import get_test_events, teardown_events


class TestEventCrud:
    @pytest.fixture(scope="function")
    def events_fixture(self, db: Session, some_experiments: List[Experiment]):
        user = get_test_user(db)
        db.query(EventEntity).filter(EventEntity.user_id == user.id).delete()
        db.flush()
        events = get_test_events(db, some_experiments)
        yield events
        teardown_events(db, events)

    def test_get_from_user(self, db: Session, events_fixture: List[EventEntity]):
        user = get_test_user(db)
        assert len(events_repo.get_to_user(db, user.id)) == len(events_fixture)

    def test_set_events_read(self, db: Session, events_fixture: List[EventEntity]):
        events_slice = events_fixture[:2]
        user = get_test_user(db)
        assert events_repo.update_read(db, events_slice, user.id) == 2
        assert events_repo.update_read(db, events_fixture, user.id) == len(
            events_fixture
        ) - len(events_slice)

    def test_get_event_by_source(self, events_fixture, db: Session):
        user = get_test_user(db)
        print(repr([evt.source for evt in events_fixture]))
        events_by_source = events_repo.get_events_by_source(db, user.id)
        print(events_by_source["training:completed"])

        def _get(source: str):
            return [evt for evt in events_fixture if evt.source == source]

        assert "training:completed" in events_by_source
        for key, evts in events_by_source.items():
            expected_len = len(_get(key))
            got_len = len(evts)
            assert (
                got_len == expected_len
            ), "got {got_len} expected {expected_len} events of source {key}"
