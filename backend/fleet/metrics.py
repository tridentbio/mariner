"""
This package provides a set of metrics for evaluating the performance of a model.
"""
import logging
from typing import Literal, Optional, Union

import numpy as np
import torch
import torchmetrics as metrics

LOG = logging.getLogger(__name__)


class Metrics:
    """
    Metrics operations that abstract the task type (regression/classification).


    Here is a table of the metrics used depending on the task type:

    | Task type | Metrics |
    | --------- | ------- |
    | regression | mse, mae, ev, mape, R2, pearson |
    | multiclass | accuracy, precision, recall, f1 |
    | multilabel | accuracy, precision, recall, f1 |



    Attributes
    """

    def __init__(
        self,
        model_type: Literal["regression", "multiclass", "multilabel"],
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
        return_type: Literal["numpy", "tensor", "float"] = "tensor",
    ):
        """Creates the metric with the used metrics.

        Args:
            model_type({Literal["regression", "multiclass", "multilabel"]}): The task type.
            num_classes: The number of classes in case the task type is classification"
            num_labels: The number of labels in case the classification is multilabel.
        """
        self.metrics = torch.nn.ModuleDict()
        self.model_type = model_type
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.return_type = return_type

        metric_names = [
            ("mse", metrics.MeanSquaredError()),
            ("mae", metrics.MeanAbsoluteError()),
            ("ev", metrics.ExplainedVariance()),
            ("mape", metrics.MeanAbsolutePercentageError()),
            ("R2", metrics.R2Score()),
            ("pearson", metrics.PearsonCorrCoef()),
        ]
        if model_type != "regression":
            assert self.num_classes is not None, "num_classes must be provided"
            kwargs = {
                "task": self.model_type,
                "num_classes": self.num_classes,
                "num_labels": self.num_labels,
            }
            metric_names = [
                ("accuracy", metrics.Accuracy(**kwargs, mdmc_reduce="global")),
                ("precision", metrics.Precision(**kwargs)),
                ("recall", metrics.Recall(**kwargs)),
                ("f1", metrics.F1Score(**kwargs)),
            ]

        for metric_name, metric in metric_names:
            self.metrics[metric_name] = metric

    def get_metrics(
        self,
        prediction: Union[np.ndarray, torch.Tensor],
        batch: Union[np.ndarray, torch.Tensor],
        mode: Literal["train", "val", "test", ""] = "",
        sufix="",
    ):
        """Gets a dict with the metrics.

        For each of the metrics available on the task type, return a dictionary
        with the metrics value given the input ``batch`` and the model output ``prediction``.

        Args:
            prediction: the model output.
            batch: object with target data (at batch['y']).
        """
        if isinstance(prediction, np.ndarray):
            prediction = torch.as_tensor(prediction)
        if isinstance(batch, np.ndarray):
            batch = torch.as_tensor(batch)
        metrics_dict = {}
        sufix = f"/{sufix}" if sufix else ""
        for metric in self.metrics:
            key = f"{mode}/{metric}{sufix}"
            try:
                if isinstance(self.metrics[metric], (metrics.Accuracy)):
                    metrics_dict[key] = self.metrics[metric](
                        prediction.squeeze().long(), batch.squeeze().long()
                    )
                else:
                    metrics_dict[key] = self.metrics[metric](
                        prediction.squeeze(), batch.squeeze()
                    )
                if self.return_type == "numpy":
                    metrics_dict[key] = metrics_dict[key].numpy()
                elif self.return_type == "float":
                    metrics_dict[key] = metrics_dict[key].item()
            except ValueError as exp:
                LOG.warning("Gor error with metric %s", metric)
                LOG.warning(exp)

        return metrics_dict

    def get_training_metrics(
        self,
        prediction: Union[np.ndarray, torch.Tensor],
        batch: Union[np.ndarray, torch.Tensor],
        sufix="",
    ):
        """Gets a dict with the training metrics.

        For each of the training metrics available on the task type, return a dictionary
        with the metrics value given the input ``batch`` and the model output ``prediction``.

        Args:
            prediction: the model output.
            batch: object with target data (at batch['y']).
        """
        return self.get_metrics(prediction, batch, mode="train", sufix=sufix)

    def get_validation_metrics(
        self,
        prediction: Union[np.ndarray, torch.Tensor],
        batch: Union[np.ndarray, torch.Tensor],
        sufix="",
    ):
        """Gets a dict with the validation metrics

        For each of the training metrics available on the task type, return a dictionary
        with the metrics value given the input ``batch`` and the model output ``prediction``

        Args:
            prediction: the model output
            batch: object with target data (at batch['y'])
        """
        return self.get_metrics(prediction, batch, mode="val", sufix=sufix)

    def get_test_metrics(
        self,
        prediction: Union[np.ndarray, torch.Tensor],
        batch: Union[np.ndarray, torch.Tensor],
        sufix="",
    ):
        """Gets a dict with the test metrics.

        For each of the test metrics available on the task type, return a dictionary
        with the metrics value given the input ``batch`` and the model output ``prediction``.

        Args:
            prediction: the model output.
            batch: object with target data (at batch['y']).
        """
        return self.get_metrics(prediction, batch, mode="test", sufix=sufix)
