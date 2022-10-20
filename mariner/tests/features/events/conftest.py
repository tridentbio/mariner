from typing import List

import pytest
from sqlalchemy.orm import Session

from app.features.experiments.schema import Experiment
from app.tests.features.events import utils


@pytest.fixture(scope="module")
def events_fixture(db: Session, some_experiments: List[Experiment]):
    return utils.events_fixture(db, some_experiments)
