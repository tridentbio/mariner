"""
This module contains the code used to manage the Ray tasks.
"""
from typing import Any, Dict, Generator, Union
from uuid import uuid4

import ray


class TaskControl:
    """
    Provides control of the running tasks in the server.


    Works as a dictionary, mapping ray ref object ids to the corresponding
    ray objects. A object ref may also be registered with some metadata,
    which is used to filter the object as needed.
    """

    def __init__(self):
        self._tasks: Dict[
            str, Union[ray.ObjectRef, ray.actor.ActorHandle]
        ] = {}
        self._tasks_metadata: Dict[str, dict] = {}

    def add_task(
        self, object_ref: ray.ObjectRef, metadata: Union[dict, None] = None
    ):
        """
        Adds a task to the control.

        Parameters:
            object_ref: The ray object ref to be added.
            metadata: The metadata to be associated with the object ref.
        """
        object_id = str(uuid4())
        self._tasks[object_id] = object_ref
        self._tasks_metadata[object_id] = metadata

    def get_task(self, object_id):
        """
        Returns the task associated with the given object id.

        Parameters:
            object_id: The object id of the task to be returned.
        """

        return self._tasks[object_id]

    def remove_task(self, object_id):
        """
        Removes the task from memory.

        Parameters:
            object_id: The object id of the task to be returned.
        """

        if object_id in self._tasks:
            self._tasks.pop(object_id)
            self._tasks_metadata.pop(object_id)

    def get_tasks(
        self, metadata=None
    ) -> tuple[list[str], list[ray.ObjectRef | ray.actor.ActorHandle]]:
        """
        Returns the tasks associated with the given metadata.

        Parameters:
            metadata: The metadata to be used to filter the tasks.
        """

        if metadata is None:
            return (
                list(self._tasks.keys()),
                [
                    self._tasks[key]
                    for key in self._tasks.keys()  # pylint: disable=C0206,C0201
                ],
            )

        tasks, ids = [], []
        for object_id, task_metadata in self._tasks_metadata.items():
            if task_metadata == metadata:
                tasks.append(self._tasks[object_id])
                ids.append(object_id)

        return ids, tasks

    def get_metadata(self, id_: str):
        """
        Returns the metadata associated with the given object id.

        Parameters:
            id_: The object id of the task to be returned.
        """
        return self._tasks_metadata.get(id_, None)

    def kill_and_remove(self, id_: str):
        """
        Kills the task and removes it from the control.

        Parameters:
            id_: The object id of the task to be returned.
        """
        if id_ in self._tasks:
            obj = self._tasks[id_]
            if isinstance(obj, ray.ObjectRef):
                ray.cancel(obj)
            else:  # assume the object is an actor
                ray.kill(obj)
            del self._tasks[id_]
            del self._tasks_metadata[id_]

    def items(self) -> Generator[tuple[str, tuple[Any, dict]], None, None]:
        """
        Returns the items of the control.

        Returns:
            A generator of the items of the control.
        """
        for id_, task in self._tasks.items():
            yield id_, (task, self._tasks_metadata[id_])


# singleton with ray task control
_task_control: Union[TaskControl, None] = None


def get_task_control() -> TaskControl:
    """
    Returns the singleton task control object.
    """
    global _task_control  # pylint: disable=global-statement
    if not _task_control:
        _task_control = TaskControl()
    return _task_control
