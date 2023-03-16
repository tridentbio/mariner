from sqlalchemy.orm import Session

from mariner.entities import Experiment as ExperimentEntity
from mariner.schemas.experiment_schemas import Experiment
from mariner.stores.experiment_sql import ExperimentUpdateRepo, experiment_store


class TestExperimentRepo:
    def test_update(self, db: Session, some_experiment: Experiment):
        target_column = some_experiment.model_version.config.dataset.target_columns[0]
        update = ExperimentUpdateRepo(
            epochs=13,
            train_metrics={
                f"train_loss_{target_column.name}": 495.88104248046875,
                "epoch": 4,
            },
            history={
                f"train_loss_{target_column.name}": [
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
        experiment_store.update(db, obj_in=update, db_obj=db_obj)
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
