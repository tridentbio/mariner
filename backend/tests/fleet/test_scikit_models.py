from typing import Literal

import pytest

from fleet import data_types, preprocessing
from fleet.dataset_schemas import DatasetConfigBuilder
from fleet.scikit_.schemas import SklearnModelSchema


class TestScikitFunctions:
    def test_model_with_preprocessing(
        self,
        numeric_to_numeric_model_config,
        numeric_to_numeric_dataset_config,
    ):
        model = numeric_to_numeric_model_config.to_model(
            numeric_to_numeric_dataset_config
        )
        assert model is not None

    @pytest.fixture
    def numeric_to_numeric_model_config(self):
        return SklearnModelSchema.parse_obj(
            {
                "model": {
                    "type": "sklearn.ensemble.RandomForestRegressor",
                    "constructorArgs": {"n_estimators": 100},
                    "fitArgs": {"X": "$stdx1", "y": "$y_regression"},
                }
            }
        )

    @pytest.fixture
    def numeric_to_numeric_dataste_config(self):
        return self.dataset_config_fixture(kind="numeric_to_numeric")

    def dataset_config_fixture(self, kind: Literal["numeric_to_numeric"]):
        return (
            DatasetConfigBuilder("test_dataset")
            .with_features(x1=data_types.NumericDataType())
            .with_targets(y_regression=data_types.NumericDataType())
            .add_transforms(
                preprocessing.StandardScalerConfig(
                    name="std-x1", forward_args={"X": "$x1"}
                )
            )
            .build()
        )
