from __future__ import annotations

import time
import uuid
from typing import Any, Dict

import ray

from .workers import PannsCpuWorker, PannsWorker


@ray.remote(num_cpus=0, max_concurrency=8, resources={"panns_worker": 0.001})
class Scheduler:
    """Small single-writer task coordinator, modeled after the reference app."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        worker_cls = PannsCpuWorker if device == "cpu" else PannsWorker
        self.worker = worker_cls.options(name="ray-panns-worker", get_if_exists=True).remote(
            checkpoint_path, device
        )
        ray.get(self.worker.warm_up.remote())
        self.tasks: Dict[str, Dict[str, Any]] = {}

    def submit(self, audio_path: str, top_k: int = 5) -> Dict[str, Any]:
        task_id = uuid.uuid4().hex
        ref = self.worker.predict.remote(audio_path, top_k)
        self.tasks[task_id] = {"state": "RUNNING", "created_at": time.time(), "ref": ref}
        return {"task_id": task_id, "state": "RUNNING"}

    def get(self, task_id: str) -> Dict[str, Any]:
        task = self.tasks.get(task_id)
        if task is None:
            return {"task_id": task_id, "state": "NOT_FOUND"}
        if task["state"] == "RUNNING":
            ready, _ = ray.wait([task["ref"]], timeout=0)
            if ready:
                try:
                    task["result"] = ray.get(ready[0])
                    task["state"] = "SUCCEEDED"
                except Exception as exc:  # surfaced as task data, not request crash
                    task["state"] = "FAILED"
                    task["error"] = str(exc)
                task["finished_at"] = time.time()
            else:
                return {"task_id": task_id, "state": "RUNNING"}
        return {k: v for k, v in task.items() if k != "ref"} | {"task_id": task_id}
