from asyncio.tasks import Task
from ctypes import ArgumentError
from typing import Any, Callable, Dict, Optional


class ExperimentView:
    def __init__(self, experiment_id: int, user_id: int, task: Task):
        self.experiment_id = experiment_id
        self.user_id = user_id
        self.task = task
        self.running_history = {}


class ExperimentManager:
    def __init__(self):
        self.experiments: Dict[int, ExperimentView] = {}

    def handle_finish(
        self,
        task: Task,
        experiment_id: int,
        done_callback: Optional[Callable[[Task, int], Any]],
    ):
        if done_callback:
            done_callback(task, experiment_id)
        self.experiments.pop(experiment_id)

    def add_experiment(
        self,
        experiment: ExperimentView,
        done_callback: Optional[Callable[[Task, int], Any]] = None,
    ):
        experiment_id = experiment.experiment_id
        task = experiment.task
        if experiment_id in self.experiments:
            raise ArgumentError("experiment_id already has a task")
        self.experiments[experiment_id] = experiment
        task.add_done_callback(
            lambda task: self.handle_finish(task, experiment_id, done_callback)
        )

    def get_running_history(self, experiment_id: int):
        if experiment_id in self.experiments:
            return self.experiments[experiment_id].running_history

    def get_from_user(self, user_id: int):
        experiments = [
            exp for exp in self.experiments.values() if exp.user_id == user_id
        ]
        return experiments

    def get_task(self, experiemnt_id: int):
        if experiemnt_id in self.experiments:
            return self.experiments[experiemnt_id].task


_task_manager: Optional[ExperimentManager] = None


def get_exp_manager():
    global _task_manager
    if not _task_manager:
        _task_manager = ExperimentManager()
    return _task_manager
