from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import mlflow
import pytest
from mlflow.tracking import MlflowClient
from pandas import DataFrame, read_csv

from fleet import fit
from fleet.base_schemas import BaseFleetModelSpec
from fleet.dataset_schemas import ColumnConfig
from fleet.model_builder import optimizers
from fleet.model_builder.schemas import TargetConfig, is_regression
from fleet.torch_.schemas import (
    MonitoringConfig,
    TorchModelSpec,
    TorchTrainingConfig,
)
from mariner.core.aws import Bucket, list_s3_objects
from tests.utils.utils import random_lower_string


@dataclass
class TestCase:
    model_spec: BaseFleetModelSpec
    train_spec: Any
    dataset_file: Union[Path, str]
    should_fail: Union[None, Any] = None
    datamodule_args = {"split_type": "random", "split_target": "60-20-20"}


def targets_head(spec: TorchModelSpec) -> TargetConfig:
    return spec.dataset.target_columns[0]


def assert_mlflow_data(spec: BaseFleetModelSpec, mlflow_experiment_id: str):
    # Checks if metrics can be found in expected mlflow location
    # and models are correctly upload to s3
    client = MlflowClient()
    run = client.search_runs(experiment_ids=[mlflow_experiment_id])[0]
    exp = client.get_experiment(mlflow_experiment_id)
    location = exp.artifact_location
    assert location.startswith(f"s3://{Bucket.Datasets.value}/{mlflow_experiment_id}")
    run_artifact_prefix = f"{mlflow_experiment_id}/{run.info.run_id}"
    objs = list_s3_objects(Bucket.Datasets, run_artifact_prefix)
    expected_artifacts = [
        f"{run_artifact_prefix}/artifacts/last/data/model.pth",
        f"{run_artifact_prefix}/artifacts/best/data/model.pth",
    ]

    object_keys = [obj for obj in objs["Contents"] if obj["Key"] in expected_artifacts]
    assert len(object_keys) == 2, "failed to trained model artifacts from s3"
    assert isinstance(
        spec, TorchModelSpec
    ), "this function is only for torch model specs"
    if is_regression(targets_head(spec)):
        for target_column in spec.dataset.target_columns:
            loss_history = client.get_metric_history(
                run_id=run.info.run_id, key=f"train/loss/{target_column.name}"
            )
            assert len(loss_history) > 0
        for target_column in spec.dataset.target_columns:
            val_loss_history = client.get_metric_history(
                run_id=run.info.run_id, key=f"val/mse/{target_column.name}"
            )
            assert len(val_loss_history) > 0
    else:
        for target_column in spec.dataset.target_columns:
            loss_history = client.get_metric_history(
                run_id=run.info.run_id, key=f"train/loss/{target_column.name}"
            )
            assert len(loss_history) > 0
        for target_column in spec.dataset.target_columns:
            val_loss_history = client.get_metric_history(
                run_id=run.info.run_id, key=f"val/precision/{target_column.name}"
            )
            assert len(val_loss_history) > 0


models_root = Path("tests") / "data" / "yaml"
datasets_root = Path("tests") / "data" / "csv"
specs = [
    (TorchModelSpec.from_yaml(models_root / path_), datasets_root / dataset_path)
    for path_, dataset_path in [
        ("dna_example.yml", "sarkisyan_full_seq_data.csv"),
        ("multiclass_classification_model.yaml", "iris.csv"),
        ("multitarget_classification_model.yaml", "iris.csv"),
        ("binary_classification_model.yaml", "iris.csv"),
        ("categorical_features_model.yaml", "zinc_extra.csv"),
        ("small_regressor_schema.yaml", "zinc.csv"),
    ]
]

test_cases = [
    TestCase(
        model_spec=spec,
        dataset_file=dataset_file,
        train_spec=TorchTrainingConfig(
            epochs=4,
            batch_size=32,
            checkpoint_config=MonitoringConfig(
                mode="min",
                metric_key=f"val/mse/{targets_head(spec).name}"
                if is_regression(targets_head(spec))
                else f"val/precision/{targets_head(spec).name}",
            ),
            optimizer=optimizers.AdamOptimizer(),
        ),
        should_fail=False,
    )
    for spec, dataset_file in specs
]


@pytest.mark.parametrize("case", test_cases)
def test_train(case: TestCase):
    mlflow_experiment_name = random_lower_string()
    mlflow_model_name = random_lower_string()
    mlflow.client.MlflowClient().create_registered_model(mlflow_model_name)
    with open(case.dataset_file) as f:
        dataset: DataFrame = read_csv(f)
        if not case.should_fail:
            result = fit(
                spec=case.model_spec,
                train_config=case.train_spec,
                dataset=dataset,
                mlflow_model_name=mlflow_model_name,
                mlflow_experiment_name=mlflow_experiment_name,
                datamodule_args=case.datamodule_args,
            )
            assert_mlflow_data(
                spec=case.model_spec, mlflow_experiment_id=result.mlflow_experiment_id
            )
        else:
            with pytest.raises(case.should_fail):
                fit(
                    spec=case.model_spec,
                    train_config=case.train_spec,
                    dataset=dataset,
                    mlflow_model_name=mlflow_model_name,
                    mlflow_experiment_name=mlflow_experiment_name,
                    datamodule_args=case.datamodule_args,
                )
