from asyncio.tasks import Task
from ctypes import ArgumentError
from typing import Any, Callable, Dict, Literal, Optional


class TaskView:
    def __init__(self, id: int, user_id: int, task: Task):
        self.id = id
        self.user_id = user_id
        self.task = task
        self.running_history = {}


class AbstractManager:
    def __init__(self):
        self.all_tasks: Dict[int, TaskView] = {}

    def handle_finish(
        self,
        task: Task,
        id: int,
        done_callback: Optional[Callable[[Task, int], Any]],
    ):
        if done_callback:
            done_callback(task, id)
        self.all_tasks.pop(id)

    def add_new_task(
        self,
        new_task: TaskView,
        done_callback: Optional[Callable[[Task, int], Any]] = None,
    ):
        id = new_task.id
        task = new_task.task
        if id in self.all_tasks:
            raise ArgumentError("id already has a task")
        self.all_tasks[id] = new_task
        task.add_done_callback(lambda task: self.handle_finish(task, id, done_callback))

    def get_running_history(self, id: int):
        if id in self.all_tasks:
            return self.all_tasks[id].running_history

    def get_from_user(self, user_id: int):
        tasks = [task for task in self.all_tasks.values() if task.user_id == user_id]
        return tasks

    def get_task(self, id: int):
        if id in self.all_tasks:
            return self.all_tasks[id].task


# TODO - adapt following TaskManager and View to use Abstracts
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


class DatasetManager(AbstractManager):
    def __init__(self):
        super().__init__()


tasks_manager: Optional[Dict[str, AbstractManager]] = {}
manager_map: Dict[str, AbstractManager] = {
    "dataset": DatasetManager,
    "experiment": ExperimentManager,
}


def get_manager(option: Literal["dataset", "experiment"]) -> AbstractManager:
    """
    Returns the manager for the given option.
    this options needs to be in the manager_map
    """
    global tasks_manager, manager_map

    if option not in tasks_manager.keys():
        tasks_manager[option] = manager_map[option]()

    return tasks_manager[option]
