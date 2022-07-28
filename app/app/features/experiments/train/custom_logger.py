from typing import Dict, List

from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from sqlalchemy.orm.session import Session

from app.db.session import SessionLocal
from app.features.experiments import controller as experiments_controller


class AppLogger(LightningLoggerBase):
    running_history: Dict[str, List[float]]
    db: Session
    experiment_id: str

    def __init__(self, experiment_id: str):
        self.db = SessionLocal()
        self.running_history = {}
        self.experiment_id = experiment_id

    @property
    def name(self):
        return "MarinerLogger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        if step:
            experiments_controller.broadcast_epoch_metrics(
                self.experiment_id, metrics, step
            )

        for metric_name, metric_value in metrics.items():
            if metric_name not in self.running_history:
                self.running_history[metric_name] = []
            self.running_history[metric_name].append(metric_value)

    @rank_zero_only
    def save(self):
        pass

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        if status != "success":
            raise Exception("training finised unsuccessfull")
        metrics = {}
        for metric_name, metric_values in self.running_history.items():
            metrics[metric_name] = metric_values[-1]

        experiments_controller.log_metrics(
            self.db,
            self.experiment_id,
            metrics,
            history=self.running_history,
            stage="train",
        )
