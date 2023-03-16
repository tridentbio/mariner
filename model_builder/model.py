"""
Torch and PytorchLightning build through from ModelSchema
"""
import logging
from typing import Dict, List, Literal, Union

import networkx as nx
import torch
import torch.nn
import torchmetrics as metrics
from pytorch_lightning.core.lightning import LightningModule

from model_builder.component_builder import AutoBuilder
from model_builder.dataset import DataInstance
from model_builder.model_schema_query import get_dependencies
from model_builder.optimizers import Optimizer
from model_builder.schemas import CategoricalDataType, ModelSchema
from model_builder.utils import collect_args


def if_str_make_list(x: Union[str, List[str]]) -> List[str]:
    """Treat something that can be either str of List[śtr] as List[str]

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
        type: Literal["regressor", "classification"],
    ):
        if type == "regressor":
            self.metrics = torch.nn.ModuleDict(
                {
                    "train_mse": metrics.MeanSquaredError(),
                    "train_mae": metrics.MeanAbsoluteError(),
                    "train_ev": metrics.ExplainedVariance(),
                    "train_mape": metrics.MeanAbsolutePercentageError(),
                    "train_R2": metrics.R2Score(),
                    "train_pearson": metrics.PearsonCorrCoef(),
                    "val_mse": metrics.MeanSquaredError(),
                    "val_mae": metrics.MeanAbsoluteError(),
                    "val_ev": metrics.ExplainedVariance(),
                    "val_mape": metrics.MeanAbsolutePercentageError(),
                    "val_R2": metrics.R2Score(),
                    "val_pearson": metrics.PearsonCorrCoef(),
                }
            )
        else:
            # https://torchmetrics.readthedocs.io/en/latest/
            self.metrics = torch.nn.ModuleDict(
                {
                    "train_accuracy": metrics.Accuracy(mdmc_reduce="global"),
                    "train_precision": metrics.Precision(),
                    "train_recall": metrics.Recall(),
                    "train_f1": metrics.F1Score(),
                    "val_accuracy": metrics.Accuracy(mdmc_reduce="global"),
                    "val_precision": metrics.Precision(),
                    "val_recall": metrics.Recall(),
                    "val_f1": metrics.F1Score(),
                }
            )

    def get_training_metrics(
        self, prediction: torch.Tensor, batch: DataInstance, sufix=""
    ):
        """Gets a dict with the training metrics

        For each of the training metrics avaliable on the task type, return a dictionary
        with the metrics value given the input ``batch`` and the model output ``prediction``

        Args:
            prediction: the model output
            batch: object with target data (at batch['y'])
        """
        metrics_dict = {}
        sufix = f"_{sufix}" if sufix else ""
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

        For each of the training metrics avaliable on the task type, return a dictionary
        with the metrics value given the input ``batch`` and the model output ``prediction``

        Args:
            prediction: the model output
            batch: object with target data (at batch['y'])
        """
        metrics_dict = {}
        sufix = f"_{sufix}" if sufix else ""
        for metric in self.metrics:
            if not metric.startswith("val"):
                continue
            metrics_dict[metric + sufix] = self.metrics[metric](
                prediction.squeeze(), batch.squeeze()
            )
        return metrics_dict


class CustomModel(LightningModule):
    """Neural network model encoded by a ModelSchema"""

    optimizer: Optimizer
    loss_dict: dict
    metrics_dict: Dict[str, Metrics]

    def __init__(self, config: Union[ModelSchema, str]):
        super().__init__()
        if isinstance(config, str):
            config = ModelSchema.parse_raw(config)
        layers_dict = {}
        self.config = config
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
        for target_column in config.dataset.target_columns:
            if not target_column.loss_fn:
                raise ValueError(f"{target_column.name}.loss_fn cannot be None")

            # This is safe as long ModelSchema was validated
            loss_fn_class = eval(target_column.loss_fn)
            self.loss_dict[target_column.name] = loss_fn_class()

            # Set up metrics for training and validation
            if target_column.column_type == "regression":
                self.metrics_dict[target_column.name] = Metrics("regressor")
            else:
                assert isinstance(target_column.data_type, CategoricalDataType)
                self.metrics_dict[target_column.name] = Metrics("classification")

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
    def call_model(model, args):
        if isinstance(args, dict):
            return model(**args)
        if isinstance(args, list):
            return model(*args)
        return model(args)

    def forward(self, input: DataInstance):  # type: ignore
        last = input
        for node_name in self.topo_sorting:
            if node_name not in self.layer_configs:
                continue  # is a featurizer, already evaluated by dataset
            if node_name in map(
                lambda x: x.out_module, self.config.dataset.target_columns
            ):
                continue  # is a target column output, evaluated separately
            layer = self.layer_configs[node_name]
            args = collect_args(input, layer.forward_args.dict())
            input[layer.name] = self.call_model(self._model[node_name], args)
            last = input[layer.name]

        result = {}
        for target_column in self.config.dataset.target_columns:
            result[target_column.name] = self.call_model(
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
            self.log(f"test_loss_{target_column.name}", loss)

        loss = sum(losses.values())
        return loss

    def validation_step(self, batch, batch_idx):
        prediction = self(batch)

        losses = {}
        for target_column in self.config.dataset.target_columns:
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
            self.log(f"val_loss_{target_column.name}", losses[target_column.name])

        loss = sum(losses.values())
        return loss

    def training_step(self, batch, batch_idx):
        prediction = self(batch)

        losses = {}
        for target_column in self.config.dataset.target_columns:
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
                    f"train_loss_{target_column.name}": losses[target_column.name],
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

        for target_column in self.config.dataset.target_columns:
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
