"""
Schemas used to specify the torch based model and how it is trained.

TODO: Move fleet.base_schema subclasses to here.
"""
import logging
from typing import Optional

from fleet.model_builder import optimizers
from fleet.model_builder.utils import CamelCaseModel

LOG = logging.getLogger(__name__)


def get_metric_mode(metric_key: str):
    """
    Returns the correct mode some metric.

    Args:
        metric_key: The key of the metric to be monitored.

    Returns:
        The mode of the metric, either "max" or "min".
    """
    metric_modes = {
        "mse": "min",
        "mae": "min",
        "ev": "max",
        "mape": "min",
        "R2": "max",
        "pearson": "max",
        "accuracy": "max",
        "precision": "max",
        "recall": "max",
        "f1": "max",
    }
    if metric_key not in metric_modes:
        LOG.warning(
            "Metric %s not found in metric_modes, defaulting to min",
            metric_key,
        )
    return metric_modes.get(metric_key, "min")


class MonitoringConfig(CamelCaseModel):
    """
    Configures model checkpointing
    """

    metric_key: str


class EarlyStoppingConfig(CamelCaseModel):
    """
    Configures earlystopping of training
    """

    metric_key: str
    min_delta: float = 5e-2
    patience: int = 5
    check_finite: bool = False


class TorchTrainingConfig(CamelCaseModel):
    """
    Configures hyperparameters in torch based model training runs.

    Attributes:
        epochs: The number of iterations on the whole training dataset.
        batch_size: The size of the batches used by the optimization
            algorithm.
        checkpoint_config: Defines the metric by which a model is considered
            superior that feeds
            :py:class:`lightning.pytorch.callbacks.ModelCheckpoint` callback.
        optimizer: The optimization algorithm described by :py:mod:`optimizers`
            class
    """

    epochs: int
    batch_size: Optional[int] = None
    checkpoint_config: Optional[MonitoringConfig] = None
    optimizer: optimizers.Optimizer = optimizers.AdamOptimizer()
    early_stopping_config: Optional[EarlyStoppingConfig] = None
    use_gpu: bool = False
