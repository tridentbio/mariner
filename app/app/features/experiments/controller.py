from typing import List

from sqlalchemy.orm.session import Session

from app.features.dataset.crud import repo as dataset_repo
from app.features.experiments.crud import ExperimentCreateRepo
from app.features.experiments.crud import repo as exp_repo
from app.features.experiments.schema import (
    Experiment,
    ListExperimentsQuery,
    TrainingRequest,
)
from app.features.experiments.train.run import start_training
from app.features.model.builder import CustomDataset
from app.features.model.crud import repo as model_repo
from app.features.model.exceptions import ModelNotFound, ModelVersionNotFound
from app.features.model.schema.model import Model
from app.features.user.model import User as UserEntity


async def create_model_traning(
    db: Session, user: UserEntity, training_request: TrainingRequest
) -> Experiment:
    model = model_repo.get_by_name(db, training_request.model_name)
    if not model:
        raise ModelNotFound()
    model = Model.from_orm(model)
    if not model or model.created_by_id != user.id:
        raise ModelNotFound()

    model_version = None
    for version in model.versions:
        if version.model_version == training_request.model_version:
            model_version = version
    if not model_version:
        raise ModelVersionNotFound()
    dataset = dataset_repo.get_by_name(db, model_version.config.dataset.name)
    torchmodel = model_version.build_torch_model()
    dataset = CustomDataset(dataset.get_dataframe(), model_version.config)
    experiment_id = await start_training(torchmodel, training_request, dataset)
    experiment = exp_repo.create(
        db,
        obj_in=ExperimentCreateRepo(
            model_name=training_request.model_name,
            model_version_name=model_version.model_version,
            experiment_id=experiment_id,
        ),
    )
    return Experiment.from_orm(experiment)


def get_experiments(
    db: Session, user: UserEntity, query: ListExperimentsQuery
) -> List[Experiment]:
    model = model_repo.get_by_name(db, query.model_name)
    if model.created_by_id != user.id:
        raise ModelNotFound()
    exps = exp_repo.get_by_model_name(db, query.model_name)
    return [Experiment.from_orm(exp) for exp in exps]
