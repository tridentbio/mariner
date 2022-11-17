from typing import List, Literal, Union

import networkx as nx
import torch
import torch.nn
import torchmetrics as metrics
from pytorch_lightning.core.lightning import LightningModule
from torch.optim.adam import Adam

from model_builder.dataset import DataInstance
from model_builder.schemas import ModelSchema
from model_builder.utils import collect_args


def if_str_make_list(x: Union[str, List[str]]) -> List[str]:
    if isinstance(x, str):
        return [x]
    return x


class Metrics:
    def __init__(self, type: Literal["regressor", "classification"]):
        if type == "regressor":
            self.metrics = torch.nn.ModuleDict(
                {
                    "train_mse": metrics.MeanSquaredError(),
                    "train_mae": metrics.MeanAbsoluteError(),
                    "train_ev": metrics.ExplainedVariance(),
                    "train_mape": metrics.MeanAbsolutePercentageError(),
                    "train_R2": metrics.R2Score(),
                    "train_pearson": metrics.PearsonCorrCoef(),
                    "train_spearman": metrics.SpearmanCorrCoef(),
                    "val_mse": metrics.MeanSquaredError(),
                    "val_mae": metrics.MeanAbsoluteError(),
                    "val_ev": metrics.ExplainedVariance(),
                    "val_mape": metrics.MeanAbsolutePercentageError(),
                    "val_R2": metrics.R2Score(),
                    "val_pearson": metrics.PearsonCorrCoef(),
                    "val_spearman": metrics.SpearmanCorrCoef(),
                }
            )
        else:
            # https://torchmetrics.readthedocs.io/en/latest/
            self.metrics = torch.nn.ModuleDict(
                {
                    "train_acc": metrics.Accuracy(),
                    "train_auroc": metrics.AUROC(),
                    "train_avgprec": metrics.AveragePrecision(),
                    "train_calibration": metrics.CalibrationError(),
                    "train_dice": metrics.Dice(),
                    "train_f1": metrics.F1Score(),
                    "train_fbeta": metrics.FBetaScore(),
                    "val_acc": metrics.Accuracy(),
                    "val_auroc": metrics.AUROC(),
                    "val_avgprec": metrics.AveragePrecision(),
                    "val_calibration": metrics.CalibrationError(),
                    "val_dice": metrics.Dice(),
                    "val_f1": metrics.F1Score(),
                    "val_fbeta": metrics.FBetaScore(),
                }
            )

    def get_training_metrics(self, prediction: torch.Tensor, batch: DataInstance):
        metrics_dict = {}
        for metric in self.metrics:
            if not metric.startswith("val"):
                continue
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
    def __init__(self, config: ModelSchema):
        super().__init__()
        layers_dict = {}
        self.config = config
        self.layer_configs = {layer.name: layer for layer in config.layers}
        for layer in config.layers:
            layer_instance = layer.create()
            layers_dict[layer.name] = layer_instance
        self._model = torch.nn.ModuleDict(layers_dict)
        self.graph = config.make_graph()
        self.topo_sorting = list(nx.topological_sort(self.graph))
        # This is safe as long ModelSchema was validated
        if not config.loss_fn:
            raise ValueError("config.loss_fn cannot be None")
        self.loss_fn = eval(config.loss_fn)

        # Set up metrics for training and validation
        if config.is_regressor():
            self.metrics = Metrics("regressor")
        elif config.is_classifier():
            self.metrics = Metrics("classification")

    def forward(self, input: DataInstance):
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
        return Adam(self.parameters(), lr=1e-3)

    def test_step(self, batch, batch_idx):
        prediction = self(batch).squeeze()
        loss = self.loss_fn(prediction, batch["y"])
        # self.logger.log('test_loss', loss)
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
