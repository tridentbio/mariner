"""
Custom pytorch lightning loggers
"""
import logging
import time
from typing import Dict, List

import requests
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from mariner.core.config import get_app_settings

LOG = logging.getLogger(__name__)


class MarinerLogger(Logger):
    """
    Custom logger that sends the metrics collected to the mariner API
    through HTTP requests. The API saves the metrics and forwards to the
    user if online.
    """

    running_history: Dict[str, List[float]]
    experiment_id: int
    experiment_name: str
    user_id: int
    last_sent_at: float = 0

    def __init__(self, experiment_id: int, experiment_name: str, user_id: int):
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
        """Creates base request payload based on instance members"""
        return {
            "experimentName": self.experiment_name,
            "experimentId": self.experiment_id,
            "userId": self.user_id,
        }

    @rank_zero_only
    def log_hyperparams(self, params):
        """Send hyperparams

        Args:
            params (Any): hyperparams to be send to the mariner API
        """
        self.send(params, "hyperparams")

    @rank_zero_only
    def log_metrics(self, metrics, step):
        """Send metrics

        Args:
            metrics (dict): dictionary with metrics
            step (int): step
        """
        data = {
            "metrics": {},
            "epoch": metrics["epoch"] if "epoch" in metrics else None,
        }
        for metric_name, metric_value in metrics.items():
            if metric_name not in self.running_history:
                self.running_history[metric_name] = []
            self.running_history[metric_name].append(metric_value)
            data["metrics"][metric_name] = metric_value
        self.send(data, "epochMetrics")

    @rank_zero_only
    def save(self):
        pass

    def send(self, msg, msg_type):
        """Sends a message to the mariner API

        Args:
            msg (dict): request payload
            msg_type (str): type of the message
        """
        data = self.make_contextualized_data()
        data["type"] = msg_type
        data["data"] = msg
        try:
            res = requests.post(
                f"{get_app_settings().SERVER_HOST}/api/v1/experiments/epoch_metrics",
                json=data,
                headers={
                    "Authorization": f"Bearer {get_app_settings().APPLICATION_SECRET}"
                },
            )
            if res.status_code != 200:
                LOG.warning(
                    "POST %s failed with status %s\n%r",
                    f"{get_app_settings().SERVER_HOST}/api/v1/experiments/epoch_metrics",
                    res.status_code,
                    res.json(),
                )

            self.last_sent_at = time.time()

        except (requests.ConnectionError, requests.ConnectTimeout):
            LOG.error(
                f"Failed metrics to {get_app_settings().SERVER_HOST}/api/v1/experiments"
                '/epoch_metrics. Make sure the env var "SERVER_HOST" is populated in '
                "the ray services, and that it points to the mariner backend"
            )
        except Exception as exp:
            LOG.error("Failed to send data from custom logger")
            LOG.error(exp)

    @rank_zero_only
    def finalize(self, status):
        if status != "success":
            raise Exception(f"training finised unsuccessfull, with status {status}")
        data = {"metrics": {}}
        for metric_name, metric_values in self.running_history.items():
            data["metrics"][metric_name] = metric_values[-1]
        data["history"] = self.running_history
        self.send(data, "metrics")
