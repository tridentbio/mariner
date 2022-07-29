from typing import Dict, List

from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from sqlalchemy.orm.session import Session

from app.db.session import SessionLocal
from app.features.experiments import controller as experiments_controller


class AppLogger(LightningLoggerBase):
    running_history: Dict[str, List[float]]
    experiment_id: str
    experiment_name: str
    user_id: int

    def __init__(self, experiment_id: str, experiment_name: str, user_id: int):
        self.running_history = {}
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self.user_id = user_id

    @property
    def name(self):
        return "MarinerLogger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        db: Session = SessionLocal()
        experiments_controller.log_hyperparams(
            db,
            self.experiment_id,
            hyperparams=params
        )
        db.close()


    @rank_zero_only
    def log_metrics(self, metrics, step):
        if step:
            experiments_controller.send_ws_epoch_update(
                self.user_id, self.experiment_id, self.experiment_name, metrics, step
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

        db: Session = SessionLocal()
        experiments_controller.log_metrics(
            db,
            self.experiment_id,
            metrics,
            history=self.running_history,
            stage="train",
        )
        db.commit()
        db.flush()
        db.close()
