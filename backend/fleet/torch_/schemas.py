from typing import Literal, Optional

from pydantic import BaseModel

from fleet.base_schemas import BaseFleetModelSpec
from fleet.model_builder import optimizers
from fleet.model_builder.schemas import TorchDatasetConfig, TorchModelSchema
from fleet.model_builder.utils import CamelCaseModel


class TorchModelSpec(BaseFleetModelSpec):
    framework: Literal["torch"] = "torch"
    spec: TorchModelSchema
    dataset: TorchDatasetConfig


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


class TorchTrainingConfig(BaseModel):
    epochs: int
    batch_size: Optional[int] = None
    checkpoint_config: Optional[MonitoringConfig] = None
    optimizer: optimizers.Optimizer = optimizers.AdamOptimizer()
    early_stopping_config: Optional[EarlyStoppingConfig] = None
