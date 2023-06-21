from typing import Union

import mlflow
import numpy as np
import pandas as pd
import sklearn.base

import fleet.mlflow
from fleet.base_schemas import BaseModelFunctions
from fleet.metrics import Metrics
from fleet.model_builder.utils import get_references_dict
from fleet.scikit_.schemas import SklearnModelSpec
from fleet.utils import data


class SciKitFunctions(BaseModelFunctions):
    """
    Operates sklearn models with functions to train, validate, test and use
    sklearn models.

    Think of this as a LightningModule from lightning, but for sklearn models. the
    `model` attribute is the sklearn model, and is None until the `train` method is
    called.

    Attributes:
        spec (SklearnModelSpec): The specification of the model.
        dataset (pd.DataFrame): The dataset to use for training, validation and testing.
        metrics (Metrics): The metrics to use for training, validation and testing.
        model (Union[None, sklearn.base.RegressorMixin, sklearn.base.ClassifierMixin]):
    """

    model: Union[None, sklearn.base.RegressorMixin, sklearn.base.ClassifierMixin] = None

    def __init__(
        self,
        spec: SklearnModelSpec,
        dataset: pd.DataFrame,
        model: Union[None, sklearn.base.ClassifierMixin] = None,
    ):
        self.spec = spec
        self.dataset = dataset
        self.metrics = Metrics(model_type="regression", return_type="float")
        self.model = model
        assert len(spec.dataset.target_columns) == 1, (
            "sklearn models only support one target column, "
            f"but got {len(spec.dataset.target_columns)}"
        )
        self.preprocessing_pipeline = data.PreprocessingPipeline(
            dataset_config=spec.dataset
        )
        if not model:  # being trained for the first time
            self._prepare_dataset(fit=True)
        else:
            self._prepare_dataset()

    def _prepare_dataset(self, fit=False):
        step = self.dataset["step"]
        X, y = self.preprocessing_pipeline.get_X_and_y(self.dataset)
        if fit:
            self.dataset = self.preprocessing_pipeline.fit_transform(X, y)
        else:
            self.dataset = self.preprocessing_pipeline.transform(X, y)
        self.dataset = self.dataset[self.preprocessing_pipeline.output_columns]
        self.dataset["step"] = step

    def _prepare_X_and_y(
        self,
        filter_step: Union[None, int] = None,
        targets=True,
    ):
        model_config = self.spec.spec

        assert self.dataset is not None, "dataset must not be None"
        if filter_step is not None:
            dataset = self.dataset[self.dataset["step"] == filter_step]

        references = get_references_dict(model_config.model.fit_args)
        args = {key: dataset.loc[:, ref] for key, ref in references.items()}
        assert "X" in args, "sklearn models take an X argument"
        X = np.stack(args["X"].to_numpy())
        if targets:
            assert "y" in args, "sklearn models take an Y argument"
            y = args["y"].to_numpy()
            return X, y
        else:
            return X

    def train(self):
        """
        Trains the sklearn model and logs the metrics on the training set to
        mlflow.
        """
        model_config = self.spec.spec
        self.model = model_config.model.create()
        if hasattr(model_config.model, "constructor_args"):
            mlflow.log_params(model_config.model.constructor_args.dict())
        X, y = self._prepare_X_and_y(filter_step=1)
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        metrics_dict = self.metrics.get_training_metrics(
            y_pred, y, sufix=self.spec.dataset.target_columns[0].name
        )
        mlflow.log_metrics(metrics_dict)
        return metrics_dict

    def val(self):
        """
        Get and log the metrics of the model from the validation set.
        """
        X, y = self._prepare_X_and_y(filter_step=2)
        if self.model is None:
            raise ValueError("sklearn model not trained")
        y_pred = self.model.predict(X)  # type: ignore
        metrics_dict = self.metrics.get_validation_metrics(
            y_pred, y, sufix=self.spec.dataset.target_columns[0].name
        )
        mlflow.log_metrics(metrics_dict)
        return metrics_dict

    def test(self):
        """
        Get and log the metrics of the model from the test set.
        """
        X, y = self._prepare_X_and_y(filter_step=3)
        if self.model is None:
            raise ValueError("sklearn model not trained")
        y_pred = self.model.predict(X)  # type: ignore
        metrics_dict = self.metrics.get_test_metrics(
            y_pred, y, sufix=self.spec.dataset.target_columns[0].name
        )
        mlflow.log_metrics(metrics_dict)
        return metrics_dict

    def predict(self, X: pd.DataFrame):
        """
        Predicts the output of the model on the given dataset.

        :todo: Fix this method, check backend/notebooks/ScikitModels.ipynb for more info.

        Args:
            X (pd.DataFrame): The dataset to predict on.
        """
        model_config = self.spec.spec
        dataset_config = self.spec.dataset
        X = self._prepare_X_and_y(targets=False)
        return self.model.predict(X)

    def log_model(
        self,
        model_name: Union[None, str] = None,
        version_description: Union[None, str] = None,
        run_id: Union[None, str] = None,
    ):
        """
        Logs the trained model to mlflow.
        """
        return fleet.mlflow.log_sklearn_model_and_create_version(
            self.model,
            model_name=model_name,
            version_description=version_description,
            run_id=run_id,
        )
