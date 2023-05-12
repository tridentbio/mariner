from typing import Optional

from fleet.model_builder import optimizers
from fleet.model_builder.utils import CamelCaseModel

# ---------------------------------- #
# Training Request Configuration     #
# ---------------------------------- #


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
    epochs: int
    batch_size: Optional[int] = None
    checkpoint_config: Optional[MonitoringConfig] = None
    optimizer: optimizers.Optimizer = optimizers.AdamOptimizer()
    early_stopping_config: Optional[EarlyStoppingConfig] = None
