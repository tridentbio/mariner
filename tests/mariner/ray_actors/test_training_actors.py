"""Tests the training actor"""
import mlflow
import mlflow.entities.model_registry.model_version
import mlflow.pytorch
import pytest
import ray
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from mlflow.tracking import MlflowClient
from sqlalchemy.orm import Session

import mariner.schemas.experiment_schemas
from mariner.core.aws import Bucket, list_s3_objects
from mariner.entities.experiment import Experiment
from mariner.ray_actors.training_actors import TrainingActor
from mariner.schemas.dataset_schemas import Dataset
from mariner.schemas.experiment_schemas import (
    EarlyStoppingConfig,
    MonitoringConfig,
    TrainingRequest,
)
from mariner.schemas.model_schemas import Model, ModelVersion
from mariner.stores.experiment_sql import ExperimentCreateRepo, experiment_store
from model_builder.model import CustomModel
from model_builder.optimizers import AdamOptimizer
from tests.fixtures.user import get_test_user
from tests.utils.utils import random_lower_string


@pytest.mark.integration
class TestTrainingActor:
    def test_train(
        self,
        db: Session,
        experiment_fixture: Experiment,
        training_request_fixture: TrainingRequest,
        mlflow_experiment_name: str,
        dataset_fixture: Dataset,
    ):
        experiment = mariner.schemas.experiment_schemas.Experiment.from_orm(
            experiment_fixture
        )
        user = get_test_user(db)
        actor = TrainingActor.remote(  # type: ignore
            experiment=experiment,
            request=training_request_fixture,
            user_id=user.id,
            mlflow_experiment_name=mlflow_experiment_name,
        )
        model: mlflow.entities.model_registry.model_version.ModelVersion = (
            ray.get(  # type:ignore
                actor.train.remote(dataset=dataset_fixture)
            )
        )
        checkpoint: ModelCheckpoint = ray.get(actor.get_checkpoint_callback.remote())  # type: ignore
        best_model_path = checkpoint.best_model_path
        last_model_path = checkpoint.last_model_path
        best_model = CustomModel.load_from_checkpoint(best_model_path)
        assert isinstance(best_model, CustomModel)
        last_model = CustomModel.load_from_checkpoint(last_model_path)
        assert isinstance(last_model, CustomModel)

        # Checks wheter model can be correctly loaded from mlflow
        # registry (by model and model version). Checks if logged models
        # are in the expected s3 artifact path, and mariner model version
        # entity is mapping to the trained mlflow model version"""
        model_name = experiment.model_version.mlflow_model_name
        version = model.version
        model_uri = f"models:/{model_name}/{version}"
        torch_model = mlflow.pytorch.load_model(model_uri)

        assert isinstance(torch_model, CustomModel)

        target_column = torch_model.config.dataset.target_columns[0]

        # Checks wheter metrics can be found in expected mlflow location
        # and models are correctly upload to s3
        client = MlflowClient()
        run = client.search_runs(experiment_ids=[experiment.mlflow_id])[0]
        exp = client.get_experiment(experiment.mlflow_id)
        location = exp.artifact_location
        assert location.startswith(
            f"s3://{Bucket.Datasets.value}/{experiment.mlflow_id}"
        )
        run_artifact_prefix = f"{experiment.mlflow_id}/{run.info.run_id}"
        objs = list_s3_objects(Bucket.Datasets, run_artifact_prefix)
        expected_artifacts = [
            f"{run_artifact_prefix}/artifacts/last/data/model.pth",
            f"{run_artifact_prefix}/artifacts/best/data/model.pth",
        ]

        object_keys = [
            obj for obj in objs["Contents"] if obj["Key"] in expected_artifacts
        ]
        assert len(object_keys) == 2, "failed to trained model artifacts from s3"
        loss_history = client.get_metric_history(
            run_id=run.info.run_id, key=f"train_loss_{target_column.name}"
        )
        assert len(loss_history) > 0
        val_loss_history = client.get_metric_history(
            run_id=run.info.run_id, key=f"val_mse_{target_column.name}"
        )
        assert len(val_loss_history) > 0

    @pytest.fixture
    def mlflow_experiment_name(self):
        return f"Testing Experiment - {random_lower_string()}"

    @pytest.fixture
    def dataset_fixture(self, some_model_integration: Model):
        return some_model_integration.dataset

    @pytest.fixture
    def modelversion_fixture(self, some_model_integration: Model) -> ModelVersion:
        return some_model_integration.versions[-1]

    @pytest.fixture(scope="function")
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
        target_column = modelversion_fixture.config.dataset.target_columns[0]
        return TrainingRequest(
            name="asdiasjd",
            model_version_id=modelversion_fixture.id,
            epochs=3,
            batch_size=32,
            checkpoint_config=MonitoringConfig(
                metric_key=f"val_mse_{target_column.name}", mode="min"
            ),
            optimizer=AdamOptimizer(),
            early_stopping_config=EarlyStoppingConfig(
                metric_key=f"val_mse_{target_column.name}", mode="min"
            ),
        )
