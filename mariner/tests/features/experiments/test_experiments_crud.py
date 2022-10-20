from sqlalchemy.orm import Session

from app.features.experiments.crud import ExperimentUpdateRepo
from app.features.experiments.crud import repo as experiment_repo
from app.features.experiments.model import Experiment as ExperimentEntity
from app.features.experiments.schema import Experiment


class TestExperimentRepo:
    def test_update(self, db: Session, some_experiment: Experiment):
        update = ExperimentUpdateRepo(
            epochs=13,
            train_metrics={"train_loss": 495.88104248046875, "epoch": 4},
            history={
                "train_loss": [
                    1349.7373046875,
                    544.4139404296875,
                    546.408447265625,
                    514.2393188476562,
                    495.88104248046875,
                ],
                "epoch": [0, 1, 2, 3, 4],
            },
        )

        db_obj = (
            db.query(ExperimentEntity)
            .filter(ExperimentEntity.id == some_experiment.id)
            .first()
        )
        experiment_repo.update(db, obj_in=update, db_obj=db_obj)
        db.commit()
        db_obj = (
            db.query(ExperimentEntity)
            .filter(ExperimentEntity.id == some_experiment.id)
            .first()
        )
        assert db_obj
        assert db_obj.train_metrics == update.train_metrics
        assert db_obj.epochs == update.epochs
        assert db_obj.history == update.history
