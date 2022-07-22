import asyncio
from asyncio.tasks import Task
from ctypes import ArgumentError
from typing import Dict, Optional

class ExperimentManager:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}

    def done_callback(self, experiment_id: str, task: Task):
        # Experiments need to have it's state updated in the database
        self.tasks.pop(experiment_id)

    def add_experiment(self, experiment_id: str, task: Task):
        if experiment_id in self.tasks:
            raise ArgumentError('experiment_id already has a task')
        self.tasks[experiment_id] = task
        task.add_done_callback(
            lambda task: self.done_callback(experiment_id, task)
        )

    def get_task(self, experiemnt_id: str):
        if experiemnt_id in self.tasks:
            return self.tasks[experiemnt_id]

_task_manager: Optional[ExperimentManager] = None
def get_exp_manager():
    global _task_manager
    if not _task_manager:
        _task_manager = ExperimentManager()
    return _task_manager

