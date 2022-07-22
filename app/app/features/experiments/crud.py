import pydantic
from sqlalchemy.orm.session import Session

from app.features.experiments.model import Experiment
from app.crud.base import CRUDBase


class ExperimentCreateRepo(pydantic.BaseModel):
    model_name: str
    model_version_name: str
    experiment_id: str
    stage: str = "NOT RUNNING"


class CRUDExperiment(CRUDBase[Experiment, ExperimentCreateRepo, ExperimentCreateRepo]):
    def get_by_model_name(self, db: Session, model_name: str):
        return db.query(Experiment).filter(Experiment.model_name == model_name).all()

repo = CRUDExperiment(Experiment)
