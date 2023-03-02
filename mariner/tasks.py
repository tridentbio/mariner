"""
Asynchronous tasks collections

Current task manager options (:see get_manager):
    - "experiment"
    - "dataset"
"""
from asyncio.tasks import Task
from ctypes import ArgumentError
from typing import Any, Callable, Dict, List, Literal, Optional


class TaskView:
    """TaskView class to type tasks.

    Attributes:
        id (int): Task id.
        user_id (int): User id.
        task (Task): Task object.
        running_history (Dict[str, Any]): Dictionary of running history.
    """

    def __init__(self, id: int, user_id: int, task: Task):
        self.id = id
        self.user_id = user_id
        self.task = task
        self.running_history = {}


class AbstractManager:
    """Abstract class for managing tasks.

    Attributes:
        all_tasks (Dict[int, TaskView]): Dictionary of all tasks being managed.
        handle_finish (Callable[[Task, int, Optional[Callable[[Task, int], Any]]]]):
            Function to be called when a task finishes.
    """

    def __init__(self):
        self.all_tasks: Dict[int, TaskView] = {}

    def handle_finish(
        self,
        task: Task,
        id: int,
        done_callback: Optional[Callable[[Task, int], Any]],
    ):
        """Function to be called when a task finishes.

        Args:
            task (Task): Task object.
            id (int): Task id.
            done_callback (Optional[Callable[[Task, int], Any]]): Callback function.
        """
        if done_callback:
            done_callback(task, id)
        self.all_tasks.pop(id)

    def add_new_task(
        self,
        new_task: TaskView,
        done_callback: Optional[Callable[[Task, int], Any]] = None,
    ):
        """Adds a new task to the manager.

        Args:
            new_task (TaskView): TaskView object.
            done_callback (Optional[Callable[[Task, int], Any]], optional):
                Callback function. Defaults to None.

        Raises:
            ArgumentError: If the task id already exists.
        """
        id = new_task.id
        task = new_task.task
        if id in self.all_tasks:
            raise ArgumentError("id already has a task")
        self.all_tasks[id] = new_task
        task.add_done_callback(lambda task: self.handle_finish(task, id, done_callback))

    def get_running_history(self, id: int) -> Dict[str, Any]:
        """Returns the running history of a task.

        Args:
            id (int): Task id.

        Returns:
            Dict[str, Any]: Dictionary of running history.
        """
        if id in self.all_tasks:
            return self.all_tasks[id].running_history

    def get_from_user(self, user_id: int) -> List[TaskView]:
        """Returns all tasks from a user.

        Args:
            user_id (int): User id.

        Returns:
            List[TaskView]: List of TaskView objects.
        """
        tasks = [task for task in self.all_tasks.values() if task.user_id == user_id]
        return tasks

    def get_task(self, id: int) -> Optional[Task]:
        """Returns the task object.

        If the task id does not exist, returns None.

        Args:
            id (int): Task id.

        Returns:
            Task: Task object.
        """
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
    """DatasetManager class."""

    def __init__(self):
        super().__init__()


# singleton instances of managers
tasks_manager: Optional[Dict[str, AbstractManager]] = {}

# map of option to manager
manager_map: Dict[str, AbstractManager] = {
    "dataset": DatasetManager,
    "experiment": ExperimentManager,
}


def get_manager(option: Literal["dataset", "experiment"]) -> AbstractManager:
    """Returns a singleton instance of manager for the given option.

    Args:
        option (str): The option for which to get the manager.
    """
    global tasks_manager, manager_map

    if option not in tasks_manager.keys():
        tasks_manager[option] = manager_map[option]()

    return tasks_manager[option]
