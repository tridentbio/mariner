from typing import Literal

import pydantic
from sqlalchemy.orm.session import Session

from app.crud.base import CRUDBase
from app.features.experiments.model import Experiment


class ExperimentCreateRepo(pydantic.BaseModel):
    model_name: str
    model_version_name: str
    experiment_id: str
    stage: str = "NOT RUNNING"


class CRUDExperiment(CRUDBase[Experiment, ExperimentCreateRepo, ExperimentCreateRepo]):
    def get_by_model_name(self, db: Session, model_name: str):
        return db.query(Experiment).filter(Experiment.model_name == model_name).all()

    def update_metrics(
        self,
        db: Session,
        experiment_id: str,
        stage: Literal["train", "val", "test"],
        metrics: dict[str, float],
        history: dict[str, list[float]],
    ):
        experiment = (
            db.query(Experiment)
            .filter(Experiment.experiment_id == experiment_id)
            .first()
        )
        experiment.history = history
        if stage == "train":
            experiment.train_metrics = metrics
        elif stage == "val":
            experiment.val_metrics = metrics
        elif stage == "test":
            experiment.test_metrics = metrics
        db.commit()
        db.flush()


repo = CRUDExperiment(Experiment)
