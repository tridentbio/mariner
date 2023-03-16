from typing import List, Literal, Optional

from sqlalchemy.orm.session import Session

from mariner.entities.experiment import Experiment as ExperimentEntity
from mariner.schemas.experiment_schemas import Experiment
from mariner.schemas.model_schemas import Model, ModelVersion
from mariner.stores.experiment_sql import ExperimentCreateRepo, experiment_store
from tests.utils.utils import random_lower_string


def mock_experiment(
    version: ModelVersion,
    user_id: int,
    stage: Optional[Literal["started", "success"]] = None,
):
    create_obj = ExperimentCreateRepo(
        epochs=1,
        experiment_name=random_lower_string(),
        mlflow_id=random_lower_string(),
        created_by_id=user_id,
        model_version_id=version.id,
    )
    if stage == "started":
        pass  # create_obj is ready
    elif stage == "success":
        target_column = version.config.dataset.target_columns[0]
        create_obj.history = {
            f"train_loss_{target_column.name}": [
                300.3,
                210.9,
                160.8,
                130.3,
                80.4,
                50.1,
                20.0,
            ]
        }
        create_obj.train_metrics = {f"train_loss_{target_column.name}": 200.3}
        create_obj.stage = "SUCCESS"
    else:
        raise NotImplementedError()
    return create_obj


def setup_experiments(db: Session, model: Model, num_experiments=4) -> List[Experiment]:
    version = model.versions[-1]
    # creates 1 started experiment and 2 successful
    exps = [
        experiment_store.create(
            db,
            obj_in=mock_experiment(
                version,
                model.created_by_id,
                stage="started" if i % 2 == 1 else "success",
            ),
        )
        for i in range(0, num_experiments)
    ]
    exps = [Experiment.from_orm(exp) for exp in exps]
    return exps


def teardown_experiments(db: Session, experiments: List[Experiment]):
    db.query(ExperimentEntity).filter(
        ExperimentEntity.id.in_([exp.id for exp in experiments])
    ).delete()
    db.flush()
