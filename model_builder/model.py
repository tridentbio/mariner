from typing import List, Literal, Optional, Union

import networkx as nx
import torch
import torch.nn
import torchmetrics as metrics
from pytorch_lightning.core.lightning import LightningModule
from torch.optim.adam import Adam

from mariner.schemas.experiment_schemas import TrainingRequest
from model_builder.dataset import DataInstance
from model_builder.layers.one_hot import OneHot
from model_builder.layers_schema import ModelbuilderonehotLayerConfig
from model_builder.optimizers import Optimizer
from model_builder.schemas import CategoricalDataType, ModelSchema
from model_builder.utils import collect_args, get_class_from_path_string


def if_str_make_list(x: Union[str, List[str]]) -> List[str]:
    if isinstance(x, str):
        return [x]
    return x


class Metrics:
    def __init__(
        self,
        type: Literal["regressor", "classification"],
        num_classes: Optional[int] = None,
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
            if not num_classes:
                raise ValueError("num_classes must be provided to classifier metrics")
            # https://torchmetrics.readthedocs.io/en/latest/
            self.metrics = torch.nn.ModuleDict(
                {
                    "train_accuracy": metrics.Accuracy(mdmc_reduce="global"),
                    "train_precision": metrics.Precision(),
                    "train_recall": metrics.Recall(),
                    "train_f1": metrics.F1Score(),
                    "train_confusion_matrix": metrics.ConfusionMatrix(
                        num_classes=num_classes
                    ),
                    "val_accuracy": metrics.Accuracy(mdmc_reduce="global"),
                    "val_precision": metrics.Precision(),
                    "val_recall": metrics.Recall(),
                    "val_f1": metrics.F1Score(),
                    "val_confusion_matrix": metrics.ConfusionMatrix(
                        num_classes=num_classes
                    ),
                }
            )

    def get_training_metrics(self, prediction: torch.Tensor, batch: DataInstance):
        metrics_dict = {}
        for metric in self.metrics:
            if not metric.startswith("train"):
                continue
            if isinstance(self.metrics[metric], (metrics.Accuracy)):
                metrics_dict[metric] = self.metrics[metric](
                    prediction.squeeze().long(), batch["y"].squeeze().long()
                )
            else:
                metrics_dict[metric] = self.metrics[metric](
                    prediction.squeeze(), batch["y"].squeeze()
                )

        return metrics_dict

    def get_validation_metrics(self, prediction: torch.Tensor, batch: DataInstance):
        metrics_dict = {}
        for metric in self.metrics:
            if not metric.startswith("val"):
                continue
            metrics_dict[metric] = self.metrics[metric](
                prediction.squeeze(), batch["y"].squeeze()
            )
        return metrics_dict


class CustomModel(LightningModule):
    """Neural network model encoded by a ModelSchema"""

    optimizer: Optimizer

    def __init__(self, config: Union[ModelSchema, str]):
        super().__init__()
        if isinstance(config, str):
            config = ModelSchema.parse_raw(config)

        layers_dict = {}
        self.config = config
        self.layer_configs = {layer.name: layer for layer in config.layers}
        for layer in config.layers:
            layer_instance = layer.create()
            if isinstance(layer_instance, OneHot) and isinstance(
                layer, ModelbuilderonehotLayerConfig
            ):
                input = layer.forward_args.x1.replace("$", "")
                for column in config.dataset.feature_columns:
                    if column.name == input and isinstance(
                        column.data_type, CategoricalDataType
                    ):
                        layer_instance.classes = column.data_type.classes
                if not layer_instance.classes:
                    raise ValueError("Failed to find OneHot classes")
            layers_dict[layer.name] = layer_instance
        self._model = torch.nn.ModuleDict(layers_dict)
        self.graph = config.make_graph()
        self.topo_sorting = list(nx.topological_sort(self.graph))
        if not config.loss_fn:
            raise ValueError("config.loss_fn cannot be None")
        # This is safe as long ModelSchema was validated
        loss_fn_class = eval(config.loss_fn)
        self.loss_fn = loss_fn_class()

        self.save_hyperparameters({"config": config.json()})
        # Set up metrics for training and validation
        if config.is_regressor():
            self.metrics = Metrics("regressor")
        elif config.is_classifier():
            assert isinstance(
                config.dataset.target_column.data_type, CategoricalDataType
            )
            self.metrics = Metrics(
                "classification",
                num_classes=len(config.dataset.target_column.data_type.classes),
            )

    def set_training_parameters(self, training_request: TrainingRequest):
        """Sets parameters that only are given in training configuration

        Parameters that are set:
            - optimizer

        Args:
            training_request:
        """
        self.optimizer = training_request.optimizer

    def forward(self, input: DataInstance):  # type: ignore
        last = input
        for node_name in self.topo_sorting:
            if node_name not in self.layer_configs:
                continue  # is a featurizer, already evaluated by dataset
            layer = self.layer_configs[node_name]
            args = collect_args(input, layer.forward_args.dict())
            if isinstance(args, dict):
                input[layer.name] = self._model[node_name](**args)
            elif isinstance(args, list):
                input[layer.name] = self._model[node_name](*args)
            else:
                input[layer.name] = self._model[node_name](args)
            last = input[layer.name]
        return last

    def configure_optimizers(self):
        if not self.optimizer:
            raise ValueError(
                "using model without setting attributes from training request"
            )
        assert self.optimizer.class_path.startswith(
            "torch.optim."
        ), "invalid start string for optimizer class_path"
        TorchOptimizer = get_class_from_path_string(self.optimizer.class_path)
        return TorchOptimizer(**self.optimizer.params.dict())

    def test_step(self, batch, batch_idx):
        prediction = self(batch).squeeze()
        loss = self.loss_fn(prediction, batch["y"])
        return loss

    def validation_step(self, batch, batch_idx):
        prediction = self(batch).squeeze()
        loss = self.loss_fn(prediction, batch["y"])
        metrics_dict = self.metrics.get_validation_metrics(prediction, batch)
        self.log_dict(
            metrics_dict, batch_size=len(batch["y"]), on_epoch=True, on_step=False
        )
        return loss

    def training_step(self, batch, batch_idx):
        prediction = self(batch)
        loss = self.loss_fn(prediction.squeeze(), batch["y"].squeeze())
        metrics_dict = {
            "train_loss": loss,
        } | self.metrics.get_training_metrics(prediction, batch)
        self.log_dict(
            metrics_dict, batch_size=len(batch["y"]), on_epoch=True, on_step=False
        )
        return loss
