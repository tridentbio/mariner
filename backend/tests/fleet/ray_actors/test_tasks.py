import time

import pytest
import ray

from fleet.ray_actors.tasks import TaskControl


@ray.remote
def long_blocking_task():
    time.sleep(2e5)
    return 42


@pytest.mark.integration
class TestTaskControl:
    task_ctl: TaskControl = TaskControl()

    def test_get_tasks(self):
        for i in range(10):
            ref = long_blocking_task.remote()
            self.task_ctl.add_task(ref, metadata={"name": f"task_{i}"})

        ids, tasks = self.task_ctl.get_tasks()
        assert len(tasks) == 10, f"Expected 10 tasks, got {len(tasks)}"
        assert len(ids) == 10, f"Expected 10 ids, got {len(ids)}"

        ids, tasks = self.task_ctl.get_tasks(metadata={"name": "task_1"})
        assert len(tasks) == 1, f"Expected 1 task, got {len(tasks)}"
        assert len(ids) == 1, f"Expected 1 id, got {len(ids)}"

    def test_cancel_tasks(self):
        ids, tasks = self.task_ctl.get_tasks()
        assert (
            len(ids) == len(tasks) == 10
        ), f"Expected 10 tasks and ids, got {len(tasks)} tasks and {len(ids)} ids"

        for id in ids:
            self.task_ctl.kill_and_remove(id)

        ids, tasks = self.task_ctl.get_tasks()
        assert (
            len(ids) == len(tasks) == 0
        ), f"Expected 0 tasks and ids, got {len(tasks)} tasks and {len(ids)} ids"
