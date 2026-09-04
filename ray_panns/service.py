import os
from typing import Any, Dict

from fastapi import Body, FastAPI, HTTPException
from ray import serve

from .scheduler import Scheduler

http_app = FastAPI(title="Ray PANNs")


@serve.deployment(
    num_replicas=1,
    ray_actor_options={
        "num_cpus": 1,
        "resources": {"panns_worker": 0.001},
    },
)
@serve.ingress(http_app)
class PannsService:
    def __init__(self):
        checkpoint = os.environ.get("PANNS_CHECKPOINT", "")
        mock = os.environ.get("PANNS_MOCK", "0").lower() in {"1", "true", "yes"}
        if not checkpoint and not mock:
            raise RuntimeError("PANNS_CHECKPOINT must point to a downloaded PANNs checkpoint")
        self.scheduler = Scheduler.options(
            name="ray-panns-scheduler", get_if_exists=True
        ).remote(checkpoint, os.getenv("PANNS_DEVICE", "cuda"))

    @http_app.get("/healthz")
    async def healthz(self) -> Dict[str, str]:
        return {"status": "ok"}

    @http_app.post("/v1/tasks")
    async def submit(self, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        audio_path = str(payload.get("audio_path", ""))
        if not audio_path:
            raise HTTPException(400, "audio_path is required")
        top_k = int(payload.get("top_k", 5))
        return await self.scheduler.submit.remote(audio_path, top_k)

    @http_app.get("/v1/tasks/{task_id}")
    async def get_task(self, task_id: str) -> Dict[str, Any]:
        return await self.scheduler.get.remote(task_id)


app = PannsService.bind()
