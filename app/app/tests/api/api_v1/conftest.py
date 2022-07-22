import pytest
from app.features.experiments.schema import Experiment, TrainingRequest

from app.features.model.schema.model import Model
from app.features.experiments.model import Experiment as ExperimentEntity
from app.features.experiments import controller as exp_ctl
from app.tests.conftest import get_test_user
from app.tests.utils.utils import random_lower_string


@pytest.fixture(scope="function")
def some_experiments(db, some_model):
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
    )  for _ in range(10) ]
    exps = [ exp_ctl.create_model_traning(db, user, request) for request in requests ]
    yield [Experiment.from_orm(exp) for exp in exps ]
    db.query(ExperimentEntity).delete()
    db.commit()

@pytest.fixture(scope="module")
def mocked_experiment_payload(some_model: Model):
    experiment_name = random_lower_string()
    version = some_model.versions[-1].model_version
    return {
        "experimentName": experiment_name,
        "learningRate": 0.05,
        "epochs": 1,
        "modelVersion": version,
        "modelName": some_model.name,
    }
