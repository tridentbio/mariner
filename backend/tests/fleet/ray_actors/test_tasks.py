import time

import ray

from fleet.ray_actors.tasks import TaskControl


@ray.remote
def long_blocking_task():
    time.sleep(2e5)
    return 42


class TestTaskControl:

    task_ctl: TaskControl = TaskControl()

    def test_get_tasks(self):
        for i in range(10):
            ref = long_blocking_task.remote()
            self.task_ctl.add_task(ref, metadata={"name": f"task_{i}"})

        tasks = self.task_ctl.get_tasks()
        assert len(tasks) == 10

        tasks = self.task_ctl.get_tasks(metadata={"name": "task_1"})
        assert len(tasks) == 1
