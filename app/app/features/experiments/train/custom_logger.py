from typing import Dict, List

import requests
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.utilities.rank_zero import rank_zero_only


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
        return "0.1"

    def make_contextualized_data(self):
        return {
            "experimentName": self.experiment_name,
            "experimentId": self.experiment_id,
            "userId": self.user_id,
        }

    @rank_zero_only
    def log_hyperparams(self, params):
        self.send(params, "hyperparams")

    @rank_zero_only
    def log_metrics(self, metrics, step):
        data = {}
        if step:
            data["step"] = step
        for metric_name, metric_value in metrics.items():
            if metric_name not in self.running_history:
                self.running_history[metric_name] = []
            self.running_history[metric_name].append(metric_value)
            data[metric_name] = metric_value
        self.send(data, "epochMetrics")

    @rank_zero_only
    def save(self):
        pass

    def send(self, msg, msg_type):
        data = self.make_contextualized_data()
        data["type"] = msg_type
        data["data"] = msg
        requests.post(
            "http://backend/api/v1/experiments/epoch_metrics", json=data
        )

    @rank_zero_only
    def finalize(self, status):
        if status != "success":
            raise Exception(f"training finised unsuccessfull, with status {status}")
        data = {"metrics": {}}
        for metric_name, metric_values in self.running_history.items():
            data["metrics"][metric_name] = metric_values[-1]
        data["history"] = self.running_history
        self.send(data, "metrics")
