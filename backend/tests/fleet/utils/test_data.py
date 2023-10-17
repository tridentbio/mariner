"""
Tests fleet.utils.data submodule.
"""
from pathlib import Path

import pandas as pd
import pytest
import torch
from pydantic import BaseModel
from torch.utils.data import Subset

from fleet import data_types
from fleet.dataset_schemas import (
    DatasetConfigBuilder,
    DatasetConfigWithPreprocessing,
)
from fleet.model_builder.constants import TrainingStep
from fleet.model_builder.splitters import apply_split_indexes
from fleet.model_schemas import FleetModelSpec
from fleet.torch_.schemas import TorchModelSpec
from fleet.utils.data import MarinerTorchDataset, PreprocessingPipeline

dataset_configs = [
    ("multiclass_classification_schema.yaml", "iris.csv"),
    ("small_regressor_schema.yaml", "zinc.csv"),
]


class TestMarinerTorchDataset:
    """
    Tests MarinerTorchDataset class.
    """

    def dataset_fixture(
        self, model="multiclass_classification_model.yaml", csv="iris.csv"
    ):
        """
        Returns MarinerTorchDataset instance.
        """
        config_path = Path.cwd() / "tests" / "data" / "yaml" / model
        dataset_path = Path.cwd() / "tests" / "data" / "csv" / csv

        config = TorchModelSpec.from_yaml(config_path)

        df = pd.read_csv(dataset_path)
        # if "step" not in df.columns:
        #     apply_split_indexes(df, split_target="60-20-20", split_type="random")
        assert "step" in df.columns, "failed to split dataset"
        keys = {}
        for c in df.columns:
            if c not in keys:
                keys[c] = 0
            keys[c] += 1
        for key, count in keys.items():
            if count >= 2:
                raise ValueError('Duplicate column name "{}"'.format(key))

        return MarinerTorchDataset(data=df, dataset_config=config.dataset)

    def test_len(self):
        """
        Tests __len__ method.
        """
        for model, csv, expected_len in zip(
            [
                "small_regressor_schema.yaml",
                "multitarget_classification_model.yaml",
                "dna_example.yml",
            ],
            [
                "zinc_extra.csv",
                "iris.csv",
                "sarkisyan_full_seq_data.csv",
            ],
            [
                20,
                20,
                20,
            ],
        ):
            dataset = self.dataset_fixture(model=model, csv=csv)
            assert len(dataset) == expected_len
            train_dataset = Subset(
                dataset, dataset.get_subset_idx(TrainingStep.TRAIN.value)
            )
            test_dataset = Subset(
                dataset, dataset.get_subset_idx(TrainingStep.TEST.value)
            )
            val_dataset = Subset(
                dataset, dataset.get_subset_idx(TrainingStep.VAL.value)
            )

            assert len(train_dataset) == pytest.approx(
                0.6 * len(dataset), rel=1
            )
            assert len(test_dataset) == pytest.approx(
                0.2 * len(dataset), rel=1
            )
            assert len(val_dataset) == pytest.approx(0.2 * len(dataset), rel=1)

    def test_lookup(self):
        """
        Tests __getitem__ method.
        """
        dataset = self.dataset_fixture()
        assert "sepal_length" in dataset[0]
        assert "sepal_width" in dataset[0]
        assert "species" in dataset[0]
        assert dataset[0]["species"].dtype == torch.int64
        assert isinstance(dataset[0]["species"], torch.Tensor)
        # assert dataset[0]["species"].shape == (1,)

    def test_get_subset_idx(self):
        """
        Tests get_subset_idx method.
        """
        df = pd.DataFrame(
            {
                "a": list(map(float, range(100))),
                "b": list(map(float, range(100))),
            }
        )
        apply_split_indexes(df, split_target="60-20-20")
        dataset_config = (
            DatasetConfigBuilder(name="Test")
            .with_features(a=data_types.NumericDataType())
            .with_targets(out_module="linear", b=data_types.NumericDataType())
            .build_torch()
        )

        dataset = MarinerTorchDataset(data=df, dataset_config=dataset_config)
        assert len(dataset) == 100
        assert len(dataset.get_subset_idx(TrainingStep.TRAIN.value)) == 60


from tests.fleet import helpers


class TestPreprocessingPipeline:
    # This class is used to resolve the FleetModelSpec union
    class ModelConfigWrapper(BaseModel):
        model_config: FleetModelSpec

    def pipeline_fixture(
        self, model_config_filename, featurize_data_types=True
    ):
        with helpers.load_test(model_config_filename) as model_config_dict:
            model_config = self.ModelConfigWrapper(
                **{"model_config": model_config_dict}
            ).model_config
            model_config.dataset = (
                model_config.dataset.to_dataset_config()
                if isinstance(
                    model_config.dataset, DatasetConfigWithPreprocessing
                )
                else model_config.dataset
            )
            return PreprocessingPipeline(
                model_config.dataset, featurize_data_types=featurize_data_types
            )

    def test_feature_leaves(self):
        cases = [
            (
                "sklearn_hiv_knearest_neighbor_classifier.yaml",
                ["MolFPFeaturizer", "OneHotEncoder"],
            ),
            (
                "sklearn_hiv_random_forest_classifier.yaml",
                ["smiles-out", "activity-out"],
            ),
        ]
        for (
            model_path,
            expected_feature_leaves,
        ) in cases:
            pipeline = self.pipeline_fixture(model_path)
            assert pipeline.features_leaves == expected_feature_leaves, (
                f"Expected feature leaves to be {', '.join(expected_feature_leaves)}, "
                f"but got {pipeline.features_leaves}"
            )
