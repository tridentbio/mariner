"""Tests the training actor"""
import mlflow
import pytest
import ray
from sqlalchemy.orm import Session

from mariner.entities.experiment import Experiment
from mariner.ray_actors.training_actors import TrainingActor
from mariner.schemas.dataset_schemas import Dataset
from mariner.schemas.experiment_schemas import MonitoringConfig, TrainingRequest
from mariner.schemas.model_schemas import Model, ModelVersion
from mariner.stores.experiment_sql import ExperimentCreateRepo, experiment_store
from model_builder.model import CustomModel
from tests.fixtures.user import get_test_user
from tests.utils.utils import random_lower_string


class TestTrainingActor:
    def test_train(
        self,
        db: Session,
        experiment_fixture: Experiment,
        dataset_fixture: Dataset,
        modelversion_fixture: ModelVersion,
        mlflow_experiment_name: str,
        training_request_fixture: TrainingRequest,
    ):
        user = get_test_user(db)
        actor = TrainingActor.remote(  # noqa
            dataset=dataset_fixture,
            modelversion=modelversion_fixture,
            request=training_request_fixture,
        )
        ray.get(
            actor.setup_loggers.remote(
                mariner_experiment=experiment_fixture,
                user_id=user.id,
                mlflow_experiment_name=mlflow_experiment_name,
            )
        )
        ray.get(actor.setup_callbacks.remote())
        model = ray.get(actor.train.remote())
        assert isinstance(
            model, CustomModel
        ), "training task does not return trained model"
        checkpoint = ray.get(actor.get_checkpoint_callback.remote())
        best_model_path = checkpoint.best_model_path
        last_model_path = checkpoint.last_model_path
        best_model = CustomModel.load_from_checkpoint(best_model_path)
        assert isinstance(best_model, CustomModel)
        last_model = CustomModel.load_from_checkpoint(last_model_path)
        assert isinstance(last_model, CustomModel)

    @pytest.mark.skip
    def test_persists_metrics(self):
        """Checks wheter metrics can be found in expected
        mlflow location (db) and mariner (db)"""

    @pytest.mark.skip
    def test_persists_model(self):
        """Checks wheter model can be correctly loaded from mlflow
        registry (by model and model version). Checks if logged models
        are in the expected s3 artifact path, and mariner model version
        entity is mapping to the trained mlflow model version"""
        ...

    @pytest.fixture
    def mlflow_experiment_name(self):
        return random_lower_string()

    @pytest.fixture
    def dataset_fixture(self, some_model: Model):
        return some_model.dataset

    @pytest.fixture
    def modelversion_fixture(self, some_model: Model) -> ModelVersion:
        return some_model.versions[-1]

    @pytest.fixture
    def experiment_fixture(
        self,
        db: Session,
        training_request_fixture: TrainingRequest,
        mlflow_experiment_name: str,
    ) -> Experiment:
        user = get_test_user(db)
        mlflow_experiment_id = mlflow.create_experiment(mlflow_experiment_name)
        experiment = experiment_store.create(
            db,
            obj_in=ExperimentCreateRepo(
                mlflow_id=mlflow_experiment_id,
                experiment_name=training_request_fixture.name,
                created_by_id=user.id,
                model_version_id=training_request_fixture.model_version_id,
                epochs=training_request_fixture.epochs,
                stage="RUNNING",
            ),
        )
        return experiment

    @pytest.fixture
    def training_request_fixture(self, modelversion_fixture: ModelVersion):
        return TrainingRequest(
            name="asdiasjd",
            model_version_id=modelversion_fixture.id,
            learning_rate=0.005,
            epochs=3,
            batch_size=32,
            checkpoint_config=MonitoringConfig(metric_key="val_mse", mode="min"),
        )
