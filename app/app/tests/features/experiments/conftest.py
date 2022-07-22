import asyncio
import pytest
from sqlalchemy.orm.session import Session
from app.features.experiments.schema import TrainingRequest
from app.features.model.schema.model import Model
from app.features.experiments.model import Experiment as ExperimentEntity
from app.features.experiments.schema import Experiment
from app.features.experiments import controller as exp_ctl 
from app.tests.conftest import get_test_user

from app.tests.utils.utils import random_lower_string

@pytest.fixture(scope="function")
async def some_experiment(db: Session, some_model: Model):
    db.query(ExperimentEntity).delete()
    db.commit()
    user = get_test_user(db)
    version = some_model.versions[-1]
    request = TrainingRequest(
        model_name=some_model.name,
        model_version=version.model_version,
        epochs=1,
        experiment_name="teste",
        learning_rate=0.05,
    )
    exp = await exp_ctl.create_model_traning(db, user, request)
    return Experiment.from_orm(exp)

@pytest.fixture(scope="function")
def some_experiments(db: Session, some_model: Model):
    db.query(ExperimentEntity).delete()
    db.commit()
    user = get_test_user(db)
    version = some_model.versions[-1]
    requests = [ TrainingRequest(
        model_name=some_model.name,
        model_version=version.model_version,
        epochs=1,
        experiment_name=random_lower_string(),
        learning_rate=0.05,
    )  for _ in range(3) ]
    exps = [ exp_ctl.create_model_traning(db, user, request) for request in requests ]
    loop = asyncio.get_event_loop()
    exps = loop.run_until_complete(asyncio.gather(*exps))
    # exps = db.query(ExperimentEntity).all()
    # exps = [Experiment.from_orm(exp) for exp in exps ]
    assert len(exps) == 3
    return exps

