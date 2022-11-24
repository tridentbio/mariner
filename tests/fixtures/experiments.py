from typing import Literal, Optional

from mariner.schemas.model_schemas import ModelVersion
from mariner.stores.experiment_sql import ExperimentCreateRepo
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
        create_obj.history = {
            "train_loss": [300.3, 210.9, 160.8, 130.3, 80.4, 50.1, 20.0]
        }
        create_obj.train_metrics = {"train_loss": 200.3}
        create_obj.stage = "SUCCESS"
    else:
        raise NotImplementedError()
    return create_obj
