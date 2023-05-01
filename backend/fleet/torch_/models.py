import logging
from typing import Dict, List, Literal, Optional, Union

import lightning.pytorch as pl
import networkx as nx
import torch
import torch.nn
import torchmetrics as metrics

from fleet.dataset_schemas import CategoricalDataType
from fleet.model_builder.component_builder import AutoBuilder
from fleet.model_builder.dataset import DataInstance
from fleet.model_builder.model_schema_query import get_dependencies
from fleet.model_builder.optimizers import Optimizer
from fleet.model_builder.schemas import TorchDatasetConfig, TorchModelSchema
from fleet.model_builder.utils import collect_args


def if_str_make_list(x: Union[str, List[str]]) -> List[str]:
    """Treat something that can be either str of List[str] as List[str]

    Converts str to single element lists

    Args:
        x: input'

    Returns:
        List[str]
    """
    if isinstance(x, str):
        return [x]
    return x


LOG = logging.getLogger(__name__)


class Metrics:
    """
    Metrics operations that abstract the task type (regression/classification)
    """

    def __init__(
        self,
        model_type: Literal["regression", "multiclass", "multilabel"],
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
    ):
        if model_type == "regression":
            self.metrics = torch.nn.ModuleDict(
                {
                    "train/mse": metrics.MeanSquaredError(),
                    "train/mae": metrics.MeanAbsoluteError(),
                    "train/ev": metrics.ExplainedVariance(),
                    "train/mape": metrics.MeanAbsolutePercentageError(),
                    "train/R2": metrics.R2Score(),
                    "train/pearson": metrics.PearsonCorrCoef(),
                    "val/mse": metrics.MeanSquaredError(),
                    "val/mae": metrics.MeanAbsoluteError(),
                    "val/ev": metrics.ExplainedVariance(),
                    "val/mape": metrics.MeanAbsolutePercentageError(),
                    "val/R2": metrics.R2Score(),
                    "val/pearson": metrics.PearsonCorrCoef(),
                }
            )
        else:
            # https://torchmetrics.readthedocs.io/en/latest/
            kwargs = {
                "task": model_type,
                "num_classes": num_classes,
                "num_labels": num_labels,
            }
            self.metrics = torch.nn.ModuleDict(
                {
                    "train/accuracy": metrics.Accuracy(**kwargs, mdmc_reduce="global"),
                    "train/precision": metrics.Precision(**kwargs),
                    "train/recall": metrics.Recall(**kwargs),
                    "train/f1": metrics.F1Score(**kwargs),
                    "val/accuracy": metrics.Accuracy(**kwargs, mdmc_reduce="global"),
                    "val/precision": metrics.Precision(**kwargs),
                    "val/recall": metrics.Recall(**kwargs),
                    "val/f1": metrics.F1Score(**kwargs),
                }
            )

    def get_training_metrics(
        self, prediction: torch.Tensor, batch: DataInstance, sufix=""
    ):
        """Gets a dict with the training metrics

        For each of the training metrics available on the task type, return a dictionary
        with the metrics value given the input ``batch`` and the model output ``prediction``

        Args:
            prediction: the model output
            batch: object with target data (at batch['y'])
        """
        metrics_dict = {}
        sufix = f"/{sufix}" if sufix else ""
        for metric in self.metrics:
            if not metric.startswith("train"):
                continue
            try:
                if isinstance(self.metrics[metric], (metrics.Accuracy)):
                    metrics_dict[metric + sufix] = self.metrics[metric](
                        prediction.squeeze().long(), batch.squeeze().long()
                    )
                else:
                    metrics_dict[metric + sufix] = self.metrics[metric](
                        prediction.squeeze(), batch.squeeze()
                    )
            except ValueError as exp:
                LOG.warning("Gor error with metric %s", metric)
                LOG.warning(exp)

        return metrics_dict

    def get_validation_metrics(
        self, prediction: torch.Tensor, batch: DataInstance, sufix=""
    ):
        """Gets a dict with the validation metrics

        For each of the training metrics available on the task type, return a dictionary
        with the metrics value given the input ``batch`` and the model output ``prediction``

        Args:
            prediction: the model output
            batch: object with target data (at batch['y'])
        """
        metrics_dict = {}
        sufix = f"/{sufix}" if sufix else ""
        for metric in self.metrics:
            if not metric.startswith("val"):
                continue
            metrics_dict[metric + sufix] = self.metrics[metric](
                prediction.squeeze(), batch.squeeze()
            )
        return metrics_dict


class CustomModel(pl.LightningModule):
    """Neural network model encoded by a ModelSchema"""

    dataset_config: TorchDatasetConfig
    optimizer: Optimizer
    loss_dict: dict
    metrics_dict: Dict[str, Metrics]

    def __init__(
        self, config: Union[TorchModelSchema, str], dataset_config: TorchDatasetConfig
    ):
        super().__init__()
        if isinstance(config, str):
            config = TorchModelSchema.parse_raw(config)
        layers_dict = {}
        self.config = config
        self.dataset_config = dataset_config
        self.layer_configs = {layer.name: layer for layer in config.layers}
        for layer in config.layers:
            layer_instance = layer.create()
            if isinstance(layer_instance, AutoBuilder):
                layer_instance.set_from_model_schema(
                    config, list(get_dependencies(layer))
                )
            layers_dict[layer.name] = layer_instance
        self._model = torch.nn.ModuleDict(layers_dict)
        self.graph = config.make_graph()
        self.topo_sorting = list(nx.topological_sort(self.graph))

        self.loss_dict = {}
        self.metrics_dict = {}
        is_multilabel = len(dataset_config.target_columns) > 1 and all(
            [
                target_column.column_type == "binary"
                for target_column in dataset_config.target_columns
            ]
        )

        for target_column in dataset_config.target_columns:
            if not target_column.loss_fn:
                raise ValueError(f"{target_column.name}.loss_fn cannot be None")

            # This is safe as long ModelSchema was validated
            loss_fn_class = eval(target_column.loss_fn)
            self.loss_dict[target_column.name] = loss_fn_class()

            # Set up metrics for training and validation
            if target_column.column_type == "regression":
                self.metrics_dict[target_column.name] = Metrics(
                    target_column.column_type
                )
            else:
                assert isinstance(target_column.data_type, CategoricalDataType)
                self.metrics_dict[target_column.name] = Metrics(
                    "multilabel" if is_multilabel else target_column.column_type,
                    num_classes=len(target_column.data_type.classes.keys()),
                    num_labels=len(dataset_config.target_columns)
                    if is_multilabel
                    else None,
                )

    def set_optimizer(self, optimizer: Optimizer):
        """Sets parameters that only are given in training configuration
        Parameters that are set:
            - optimizer
        Args:
            training_request:
        """
        self.optimizer = optimizer

        self.save_hyperparameters(
            {"config": self.config.json(), "learning_rate": self.optimizer.params.lr}
        )

    @staticmethod
    def _call_model(model, args):
        if isinstance(args, dict):
            return model(**args)
        if isinstance(args, list):
            return model(*args)
        return model(args)

    def forward(self, input_: DataInstance):  # type: ignore
        last = input_
        for node_name in self.topo_sorting:
            if node_name not in self.layer_configs:
                continue  # is a featurizer, already evaluated by dataset
            if node_name in map(
                lambda x: x.out_module, self.dataset_config.target_columns
            ):
                continue  # is a target column output, evaluated separately
            layer = self.layer_configs[node_name]
            args = collect_args(input_, layer.forward_args.dict())
            input_[layer.name] = CustomModel._call_model(self._model[node_name], args)
            last = input_[layer.name]

        result = {}
        for target_column in self.dataset_config.target_columns:
            result[target_column.name] = self._call_model(
                self._model[target_column.out_module], last
            )

        return result

    def configure_optimizers(self):
        if not self.optimizer:
            raise ValueError(
                "using model without setting attributes from training request"
            )
        assert self.optimizer.class_path.startswith(
            "torch.optim."
        ), "invalid start string for optimizer class_path"
        return self.optimizer.create(self.parameters())

    def test_step(self, batch, batch_idx):
        predictions = self(batch)

        losses = {}
        for target_column in self.config.dataset.target_columns:
            args = (
                predictions[target_column.name].squeeze(),
                batch[target_column.name],
            )
            losses[target_column.name] = self.loss_dict[target_column.name](*args)
            self.log(f"test/loss/{target_column.name}", losses[target_column.name])

        loss = sum(losses.values())
        return loss

    def validation_step(self, batch, batch_idx):
        prediction = self(batch)

        losses = {}
        for target_column in self.dataset_config.target_columns:
            args = (
                prediction[target_column.name].squeeze(),
                batch[target_column.name].squeeze(),
            )

            if target_column.column_type != "multiclass":
                args = map(lambda x: x.type(torch.FloatTensor), args)

            losses[target_column.name] = self.loss_dict[target_column.name](*args)
            metrics_dict = self.metrics_dict[target_column.name].get_validation_metrics(
                prediction[target_column.name],
                batch[target_column.name],
                target_column.name,
            )
            self.log_dict(
                metrics_dict,
                batch_size=len(batch[target_column.name]),
                on_epoch=True,
                on_step=False,
            )
            self.log(f"val/loss/{target_column.name}", losses[target_column.name])

        loss = sum(losses.values())
        return loss

    def training_step(self, batch, batch_idx):
        prediction = self(batch)

        losses = {}
        for target_column in self.dataset_config.target_columns:
            args = (
                prediction[target_column.name].squeeze(),
                batch[target_column.name].squeeze(),
            )

            if target_column.column_type != "multiclass":
                args = map(lambda x: x.type(torch.FloatTensor), args)

            losses[target_column.name] = self.loss_dict[target_column.name](*args)
            metrics_dict = self.metrics_dict[target_column.name].get_training_metrics(
                prediction[target_column.name],
                batch[target_column.name],
                target_column.name,
            )
            self.log_dict(
                {
                    **metrics_dict,
                    f"train/loss/{target_column.name}": losses[target_column.name],
                },
                batch_size=len(batch[target_column.name]),
                on_epoch=True,
                on_step=False,
            )

        loss = sum(losses.values())
        return loss

    def predict_step(self, batch, batch_idx=0):
        """
        Runs the forward pass using self.forward, then, for
        classifier models, normalizes the predictions using a
        non-linearity like sigmoid or softmax.
        """
        predictions = self(batch)

        for target_column in self.dataset_config.target_columns:
            if target_column.column_type == "multiclass":
                predictions[target_column.name] = torch.nn.functional.softmax(
                    predictions[target_column.name].squeeze(), dim=-1
                )
            elif target_column.column_type == "binary":
                predictions[target_column.name] = torch.sigmoid(
                    predictions[target_column.name].squeeze()
                )
            else:
                predictions[target_column.name] = predictions[
                    target_column.name
                ].squeeze()

        return predictions
