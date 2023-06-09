"""
Trainable lightning model classes and is mapped from :py:class:`fleet.torch_.schemas.TorchModelSpec`
as well as auxiliary classes and functions.
"""
import logging
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

import lightning.pytorch as pl
import networkx as nx
import torch
import torch.nn

from fleet.metrics import Metrics
from fleet.model_builder.component_builder import AutoBuilder
from fleet.model_builder.dataset import DataInstance
from fleet.model_builder.model_schema_query import get_dependencies
from fleet.model_builder.optimizers import Optimizer
from fleet.model_builder.schemas import TorchModelSchema
from fleet.model_builder.utils import collect_args

if TYPE_CHECKING:
    from fleet.dataset_schemas import TorchDatasetConfig


def if_str_make_list(x: Union[str, List[str]]) -> List[str]:
    """Treat something that can be either str of List[str] as List[str]

    Converts str to single element lists

    Args:
        x: input

    Returns:
        List[str]
    """
    if isinstance(x, str):
        return [x]
    return x


LOG = logging.getLogger(__name__)


def _adapt_loss_args(
    loss_fn: type, args: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    # CrossEntropyLoss requires targets to be longs
    if isinstance(
        loss_fn,
        (torch.nn.CrossEntropyLoss),
    ):
        return (args[0], args[1].long())

    # BCEWithLogitsLoss requires targets to be floats
    elif isinstance(
        loss_fn,
        (torch.nn.BCEWithLogitsLoss),
    ):
        return (args[0], args[1].float())
    return args


class CustomModel(pl.LightningModule):
    """Neural network model encoded by a ModelSchema"""

    dataset_config: "TorchDatasetConfig"
    optimizer: Optimizer
    loss_dict: dict
    metrics_dict: Dict[str, Metrics]

    def __init__(
        self,
        config: Union[TorchModelSchema, str],
        dataset_config: "TorchDatasetConfig",
    ):
        super().__init__()
        if isinstance(config, str):
            config = TorchModelSchema.parse_raw(config)
        self.save_hyperparameters({"config": config.json()})
        layers_dict = {}
        self.config = config
        self.layer_configs = {layer.name: layer for layer in config.layers}
        for layer in config.layers:
            layer_instance = layer.create()
            if isinstance(layer_instance, AutoBuilder):
                layer_instance.set_from_model_schema(
                    dataset_config=dataset_config,
                    config=config,
                    deps=list(get_dependencies(layer)),
                )
            layers_dict[layer.name] = layer_instance

        self.dataset_config = dataset_config
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
                self.metrics_dict[target_column.name] = Metrics("regression")
            else:
                self.metrics_dict[target_column.name] = Metrics(
                    "multilabel" if is_multilabel else "multiclass",
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

    @staticmethod
    def _call_model(model, args):
        if isinstance(args, dict):
            return model(**args)
        if isinstance(args, list):
            return model(*args)
        return model(args)

    def forward(self, input_: DataInstance):  # type: ignore
        """Forward pass of the model

        Runs the foward pass through the layers ordered by the topological
        sorting on self.topo_sorting attribute.
        If it finds a featurizer, ignores because it already evaluated by the
        dataset.
        If it finds a target column output (out_module), ignores on topo_sorting
        iteration then evaluate this for
        each target column on self.config.dataset.target_columns iteration.

        Args:
            input_: input data (x)
        Returns:
            dictionary with the output layers outputs
        """
        out_layers = map(lambda x: x.out_module, self.dataset_config.target_columns)
        for node_name in self.topo_sorting:
            if node_name not in self.layer_configs:
                continue  # is a featurizer, already evaluated by dataset
            if node_name in out_layers:
                continue  # is a target column output, evaluated separately
            layer = self.layer_configs[node_name]
            args = collect_args(input_, layer.forward_args.dict())
            input_[layer.name] = CustomModel._call_model(self._model[node_name], args)

        result = {}
        for target_column in self.dataset_config.target_columns:
            layer = self.layer_configs[target_column.out_module]
            args = collect_args(input_, layer.forward_args.dict())
            result[target_column.name] = self._call_model(
                self._model[target_column.out_module], args
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

    def test_step(self, batch, _batch_idx):
        predictions = self(batch)

        losses = {}
        for target_column in self.dataset_config.target_columns:
            args = (
                predictions[target_column.name].squeeze(),
                batch[target_column.name].squeeze(),
            )

            loss_fn = self.loss_dict[target_column.name]
            args = _adapt_loss_args(loss_fn, args)

            losses[target_column.name] = loss_fn(*args)
            self.log(f"test/loss/{target_column.name}", losses[target_column.name])

        loss = sum(losses.values())
        return loss

    def validation_step(self, batch, _batch_idx):
        prediction: Dict[str, torch.Tensor] = self(batch)

        losses = {}
        for target_column in self.dataset_config.target_columns:
            args = (
                prediction[target_column.name].squeeze(),
                batch[target_column.name].squeeze(),
            )

            loss_fn = self.loss_dict[target_column.name]

            args = _adapt_loss_args(loss_fn, args)
            losses[target_column.name] = loss_fn(*args)
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

    def training_step(self, batch, _batch_idx):
        prediction: Dict[str, torch.Tensor] = self(batch)

        losses = {}
        for target_column in self.dataset_config.target_columns:

            args = (
                prediction[target_column.name].squeeze(),
                batch[target_column.name].squeeze(),
            )
            loss_fn = self.loss_dict[target_column.name]
            args = _adapt_loss_args(loss_fn, args)

            losses[target_column.name] = loss_fn(*args)
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

    def predict_step(self, batch, _batch_idx=0, dataloader_idx=0):
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
