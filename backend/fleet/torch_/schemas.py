"""
Schemas used to specify the torch based model and how it is trained.

TODO: Move fleet.base_schema subclasses to here.
"""
from typing import Optional

from fleet.model_builder import optimizers
from fleet.model_builder.utils import CamelCaseModel


class MonitoringConfig(CamelCaseModel):
    """
    Configures model checkpointing
    """

    metric_key: str
    mode: str


class EarlyStoppingConfig(CamelCaseModel):
    """
    Configures earlystopping of training
    """

    metric_key: str
    mode: str
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
