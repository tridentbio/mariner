"""
Functions to train and use sklearn models.
"""
import logging
from typing import Dict, List, Literal, Union

import mlflow
import numpy as np
import pandas as pd
import sklearn.base

import fleet.mlflow
from fleet.base_schemas import BaseModelFunctions
from fleet.dataset_schemas import is_regression_column
from fleet.metrics import Metrics
from fleet.model_builder.constants import TrainingStep
from fleet.model_builder.utils import get_references_dict
from fleet.scikit_.schemas import SklearnModelSpec
from fleet.utils import data

LOG = logging.getLogger(__name__)


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

    model: Union[
        None, sklearn.base.RegressorMixin, sklearn.base.ClassifierMixin
    ] = None
    data: Dict[str, Union[pd.Series, np.ndarray, List[np.ndarray]]] = None

    def __init__(
        self,
        spec: SklearnModelSpec,
        dataset: pd.DataFrame | None = None,
        model: Union[None, sklearn.base.ClassifierMixin] = None,
        preprocessing_pipeline: Union[
            None, "data.PreprocessingPipeline"
        ] = None,
    ):
        self.spec = spec
        self.dataset = dataset
        self.data = {}
        assert isinstance(
            spec.dataset, data.DatasetConfig
        ), "dataset must be converted to DatasetConfig previously"
        model_type = (
            "regression"
            if is_regression_column(spec.dataset)
            else "multiclass"
        )
        num_classes = None
        if model_type == "multiclass":
            num_classes = len(spec.dataset.target_columns[0].data_type.classes)

        self.metrics = Metrics(
            model_type=model_type,
            num_classes=num_classes,
            return_type="float",
        )
        self.model = model
        assert len(spec.dataset.target_columns) == 1, (
            "sklearn models only support one target column, "
            f"but got {len(spec.dataset.target_columns)}"
        )
        if preprocessing_pipeline:
            self.preprocessing_pipeline = preprocessing_pipeline
        else:
            self.preprocessing_pipeline = data.PreprocessingPipeline(
                dataset_config=spec.dataset
            )
        self.references = get_references_dict(
            spec.spec.model.fit_args
            or {"X": "$dataset-feats-out", "y": "$dataset-targets-out"}
        )

    def _filter_data(
        self,
        filtering: pd.Series,
        data_: Union[pd.Series, np.ndarray, List[np.ndarray]],
    ):
        if isinstance(data_, (pd.DataFrame, pd.Series, np.ndarray)):
            return data_[filtering]
        elif isinstance(data_, list):
            return [
                data_[i]
                for i, filter_value in enumerate(filtering)
                if filter_value
            ]

    def _prepare_X_and_y(
        self,
        filter_step: Union[None, int] = None,
        targets=True,
    ):
        assert (
            self.data
        ), "_prepare_data must be called before calling _prepare_X_and_y"
        assert self.dataset is not None, "dataset must not be None"

        filtering = None
        if filter_step:
            filtering = self.dataset["step"] == filter_step

        args = self.data[self.references["X"]], self.data[
            self.references["y"]
        ] if targets else (self.data[self.references["X"]],)

        if filtering is not None:
            args = map(lambda x: self._filter_data(filtering, x), args)

        return args

    def _prepare_data(self, fit=False, only_features=False):
        X, y = self.preprocessing_pipeline.get_X_and_y(self.dataset)
        if fit:
            self.data = self.preprocessing_pipeline.fit_transform(
                X, y, concat_features=True, concat_targets=True
            )
        else:
            self.data = self.preprocessing_pipeline.transform(
                X,
                y,
                concat_features=True,
                concat_targets=True,
                only_features=only_features,
            )

    def get_metrics(
        self,
        prediction: np.ndarray,
        batch: Union[np.ndarray, pd.Series],
        sufix="",
        step: Literal["train", "val", "test"] = "train",
    ) -> dict:
        """
        Returns the metrics for the given prediction and batch.
        """
        if isinstance(batch, pd.Series):
            # converts to numpy and cast to float if it's long
            batch = batch.to_numpy()

        if batch.dtype == np.int64:
            batch = batch.astype(np.float32)
        if prediction.dtype == np.int64:
            prediction = prediction.astype(np.float32)

        if step == "train":
            return self.metrics.get_training_metrics(
                prediction, batch, sufix=sufix
            )
        elif step == "val":
            return self.metrics.get_validation_metrics(
                prediction, batch, sufix=sufix
            )
        elif step == "test":
            return self.metrics.get_test_metrics(
                prediction, batch, sufix=sufix
            )

    def train(self):
        """
        Trains the sklearn model and logs the metrics on the training set to
        mlflow.
        """
        self._prepare_data(fit=True)
        model_config = self.spec.spec
        self.model = model_config.model.create()
        if hasattr(model_config.model, "constructor_args"):
            mlflow.log_params(model_config.model.constructor_args.dict())
        X, y = self._prepare_X_and_y(filter_step=TrainingStep.TRAIN.value)
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        metrics_dict = self.get_metrics(
            y_pred,
            y,
            sufix=self.spec.dataset.target_columns[0].name,
            step="train",
        )
        mlflow.log_metrics(metrics_dict)
        return metrics_dict

    def val(self):
        """
        Get and log the metrics of the model from the validation set.
        """
        self._prepare_data(fit=False)
        X, y = self._prepare_X_and_y(filter_step=TrainingStep.VAL.value)
        if self.model is None:
            raise ValueError("sklearn model not trained")
        y_pred = self.model.predict(X)  # type: ignore
        metrics_dict = self.get_metrics(
            y_pred,
            y,
            sufix=self.spec.dataset.target_columns[0].name,
            step="val",
        )
        mlflow.log_metrics(metrics_dict)
        return metrics_dict

    def test(self):
        """
        Get and log the metrics of the model from the test set.
        """
        self._prepare_data(fit=False)
        X, y = self._prepare_X_and_y(filter_step=TrainingStep.TEST.value)
        if self.model is None:
            raise ValueError("sklearn model not trained")
        y_pred = self.model.predict(X)  # type: ignore
        metrics_dict = self.get_metrics(
            y_pred,
            y,
            sufix=self.spec.dataset.target_columns[0].name,
            step="test",
        )
        mlflow.log_metrics(metrics_dict)
        return metrics_dict

    def predict(
        self,
        input_: pd.DataFrame,
    ):
        """
        Predicts the output of the model on the given dataset.

        Args:
            input_ (pd.DataFrame): The dataset to predict on.
        """
        task_type = self.spec.spec.model.task_type
        is_classification = ("multiclass" in task_type) or (
            "multilabel" in task_type
        )

        transformed_data = self.preprocessing_pipeline.transform(
            input_, concat_features=True, only_features=True
        )
        X = transformed_data[self.references["X"]]

        target_column = self.spec.dataset.target_columns[0]
        if is_classification:
            prediction_probs = self.model.predict_proba(X)
            col_featurizer_config = self.spec.dataset.get_featurizer(
                target_column.name
            )

            if (
                col_featurizer_config is not None
                and "OneHotEncoder" in col_featurizer_config.type
            ):
                print("Getting prediction")
                prediction_probs = prediction_probs[0].tolist()
                print("Getting classes")
                classes = (
                    self.preprocessing_pipeline.featurizers[
                        col_featurizer_config.name
                    ][0]
                    .categories_[0]
                    .tolist()
                )
            elif (
                col_featurizer_config is not None
                and "LabelEncoder" in col_featurizer_config.type
            ):
                prediction_probs = prediction_probs.tolist()
                classes = self.preprocessing_pipeline.featurizers[
                    col_featurizer_config.name
                ][0].classes_.tolist()

            else:
                LOG.warning(
                    "Classifier %s was not using any encoder", self.spec.name
                )
                classes = list(
                    map(
                        lambda entry: entry[0],
                        sorted(target_column.data_type.classes.items()),
                    )
                )
            return {
                target_column.name: {
                    "probs": prediction_probs,
                    "classes": classes,
                }
            }
        else:
            predictions = self.model.predict(X)
            return {target_column.name: predictions.tolist()}

    def log_models(
        self,
        mlflow_model_name: Union[None, str] = None,
        version_description: Union[None, str] = None,
        run_id: Union[None, str] = None,
    ):
        """
        Logs the trained model to mlflow.
        """
        return fleet.mlflow.log_sklearn_model_and_create_version(
            self.model,
            model_name=mlflow_model_name,
            version_description=version_description,
            run_id=run_id,
        )
