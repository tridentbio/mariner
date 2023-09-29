"""
Tests for fleet.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from fleet.base_schemas import FleetModelSpec, TorchModelSpec
from fleet.dataset_schemas import (
    TargetTorchColumnConfig,
    is_regression,
    is_regression_column,
)
from fleet.model_builder import optimizers
from fleet.model_builder.splitters import apply_split_indexes
from fleet.model_functions import fit, predict, run_test
from fleet.scikit_.schemas import SklearnModelSpec
from fleet.torch_.schemas import MonitoringConfig, TorchTrainingConfig
from mariner.core.aws import Bucket, list_s3_objects
from tests.utils.utils import random_lower_string


@dataclass
class TestCase:
    model_spec: FleetModelSpec
    train_spec: Any
    dataset_file: Union[Path, str]
    predict_sample: Union[None, dict] = None
    should_fail: Union[None, Any] = None
    datamodule_args = {"split_type": "random", "split_target": "60-20-20"}


def targets_head(spec: TorchModelSpec) -> "TargetTorchColumnConfig":
    return spec.dataset.target_columns[0]


def assert_mlflow_metric(client: MlflowClient, run_id: str, key: str):
    val_metric = client.get_metric_history(run_id=run_id, key=key)
    assert len(val_metric) > 0, f"failed to save metric {key} failed"


def assert_mlflow_data(
    spec: FleetModelSpec,
    mlflow_experiment_id: str,
    framework="torch",
):
    """
    Checks if metrics can be found in expected mlflow location
    and models are correctly upload to s3
    """
    client = MlflowClient()
    run = client.search_runs(experiment_ids=[mlflow_experiment_id])[0]
    exp = client.get_experiment(mlflow_experiment_id)
    location = exp.artifact_location
    assert location.startswith(
        f"s3://{Bucket.Datasets.value}/{mlflow_experiment_id}"
    )
    run_artifact_prefix = f"{mlflow_experiment_id}/{run.info.run_id}"
    objs = list_s3_objects(Bucket.Datasets, run_artifact_prefix)

    if spec.framework == "torch":
        expected_artifacts = [
            f"{run_artifact_prefix}/artifacts/last/data/model.pth",
            f"{run_artifact_prefix}/artifacts/best/data/model.pth",
        ]
        object_keys = [
            obj for obj in objs["Contents"] if obj["Key"] in expected_artifacts
        ]
        assert len(object_keys) == len(
            expected_artifacts
        ), "failed to find trained model artifacts from s3"

    elif spec.framework == "sklearn":
        expected_artifacts = [
            f"{run_artifact_prefix}/artifacts/model/model.pkl"
        ]
        object_keys = [
            obj for obj in objs["Contents"] if obj["Key"] in expected_artifacts
        ]
        assert len(object_keys) == len(
            expected_artifacts
        ), "failed to trained model from s3"

    # TODO: check pipeline transform persistence
    for target_column in spec.dataset.target_columns:
        if spec.framework == "torch":
            loss_history = client.get_metric_history(
                run_id=run.info.run_id, key=f"train/loss/{target_column.name}"
            )
            assert len(loss_history) > 0

        assert_mlflow_metric(
            client=client,
            run_id=run.info.run_id,
            key=f"val/mse/{target_column.name}"
            if is_regression_column(spec.dataset, target_column.name)
            else f"val/precision/{target_column.name}",
        )


predict_samples = {
    "hiv": {"smiles": ["CCCC"], "activity": ["CI"]},
    "hiv2": {"smiles": ["CCCC", "CCCCCC"], "activity": ["CM", "CI"]},
    "sampl": {
        "smiles": ["CCCC"],
    },
    "sampl2": {
        "smiles": ["CCCC", "CCCCCC"],
    },
    "iris_binary": {"sepal_length": [1.6], "sepal_width": [0.9]},
}


models_root = Path("tests") / "data" / "yaml"
datasets_root = Path("tests") / "data" / "csv"
specs = [
    (
        TorchModelSpec.from_yaml(models_root / model_file)
        if not model_file.startswith("sklearn")
        else SklearnModelSpec.from_yaml(models_root / model_file),
        datasets_root / dataset_file,
        predict_samples.get(predict_sample_key),
    )
    for model_file, dataset_file, predict_sample_key in [
        ("binary_classification_model.yaml", "iris.csv", "iris_binary"),
        (
            "sklearn_sampl_random_forest_regressor.yaml",
            "SAMPL.csv",
            "sampl",
        ),
        ("sklearn_sampl_knn_regressor.yaml", "SAMPL.csv", "sampl"),
        (
            "sklearn_sampl_extra_trees_regressor.yaml",
            "SAMPL.csv",
            "sampl2",
        ),
        (
            "sklearn_sampl_knearest_neighbor_regressor.yaml",
            "SAMPL.csv",
            "sampl2",
        ),
        ("sklearn_hiv_extra_trees_classifier.yaml", "HIV.csv", "hiv"),
        (
            "sklearn_hiv_knearest_neighbor_classifier.yaml",
            "HIV.csv",
            "hiv",
        ),
        ("sklearn_hiv_random_forest_classifier.yaml", "HIV.csv", "hiv2"),
        ("small_regressor_schema.yaml", "zinc.csv", ""),
        ("multiclass_classification_model.yaml", "iris.csv", ""),
        ("multitarget_classification_model.yaml", "iris.csv", ""),
        ("dna_example.yml", "sarkisyan_full_seq_data.csv", ""),
    ]
]
test_cases = [
    TestCase(
        model_spec=spec,
        dataset_file=dataset_file,
        predict_sample=predict_sample,
        train_spec=TorchTrainingConfig(
            epochs=4,
            batch_size=32,
            checkpoint_config=MonitoringConfig(
                metric_key=f"val/mse/{targets_head(spec).name}"
                if is_regression(targets_head(spec))
                else f"val/precision/{targets_head(spec).name}",
            ),
            optimizer=optimizers.AdamOptimizer(),
        )
        if spec.framework == "torch"
        else None,
        should_fail=False,
    )
    for spec, dataset_file, predict_sample in specs
]


@pytest.mark.parametrize("case", test_cases)
@pytest.mark.integration
def test_train(case: TestCase):
    """
    Tests if training works with mlflow
    """
    mlflow_experiment_name = random_lower_string()
    mlflow_model_name = random_lower_string()
    mlflow.client.MlflowClient().create_registered_model(mlflow_model_name)
    with open(case.dataset_file, encoding="utf-8") as file:
        dataset: pd.DataFrame = pd.read_csv(file)
        if "step" not in dataset.columns:
            apply_split_indexes(dataset)
        assert "step" in dataset
        if not case.should_fail:
            result = fit(
                spec=case.model_spec,
                train_config=case.train_spec,
                dataset=dataset.copy(),
                mlflow_model_name=mlflow_model_name,
                mlflow_experiment_name=mlflow_experiment_name,
                datamodule_args=case.datamodule_args,
            )
            assert (
                result.mlflow_model_version is not None
            ), "model version is None"
            assert_mlflow_data(
                spec=case.model_spec,
                mlflow_experiment_id=result.mlflow_experiment_id,
            )
            run_test(
                spec=case.model_spec,
                mlflow_model_name=mlflow_model_name,
                mlflow_model_version=result.mlflow_model_version.version,
                dataset=dataset,
            )

            if case.predict_sample:
                result = predict(
                    spec=case.model_spec,
                    mlflow_model_name=mlflow_model_name,
                    mlflow_model_version=result.mlflow_model_version.version,
                    input_=case.predict_sample,
                )
                assert len(
                    result[case.model_spec.dataset.target_columns[0].name]
                ) == len(
                    next(iter(case.predict_sample.values()))
                ), "prediction result needs to have the same length as the input"

        else:
            with pytest.raises(case.should_fail):
                result = fit(
                    spec=case.model_spec,
                    train_config=case.train_spec,
                    dataset=dataset,
                    mlflow_model_name=mlflow_model_name,
                    mlflow_experiment_name=mlflow_experiment_name,
                    datamodule_args=case.datamodule_args,
                )
