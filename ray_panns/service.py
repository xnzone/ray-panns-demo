"""HTTP producer deployment for the Ray Serve async-inference pipeline."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import Body, FastAPI, HTTPException
from ray import serve
from ray.serve.task_consumer import instantiate_adapter_from_config

from .config import PROCESSOR_CONFIG, TASK_PREDICT_AUDIO
from .workers import PannsConsumer, PannsEncoder

http_app = FastAPI(title="Ray PANNs Async Inference")


def _task_result(result: Any) -> Dict[str, Any]:
    """Convert Ray's pydantic TaskResult to stable JSON across pydantic versions."""
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return {"id": result.id, "status": result.status, "created_at": result.created_at, "result": result.result}


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 1, "resources": {"panns_worker": 0.001}},
)
@serve.ingress(http_app)
class PannsIngress:
    """Fast producer: enqueue work and never wait for model execution."""

    def __init__(self, consumer):
        # Keep the handle in the graph so Serve starts the consumer and encoder.
        self.consumer = consumer
        self.adapter = instantiate_adapter_from_config(PROCESSOR_CONFIG)

    @http_app.get("/healthz")
    async def healthz(self) -> Dict[str, str]:
        return {"status": "ok", "queue": PROCESSOR_CONFIG.queue_name}

    @http_app.post("/v1/tasks")
    async def submit(self, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        audio_path = str(payload.get("audio_path", "")).strip()
        if not audio_path:
            raise HTTPException(400, "audio_path is required")
        try:
            top_k = max(1, min(int(payload.get("top_k", 5)), 100))
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, "top_k must be an integer") from exc

        result = self.adapter.enqueue_task_sync(
            task_name=TASK_PREDICT_AUDIO,
            kwargs={"audio_path": audio_path, "top_k": top_k},
        )
        response = _task_result(result)
        response["status_url"] = f"/v1/tasks/{response['id']}"
        response["task_id"] = response.pop("id")
        return response

    @http_app.get("/v1/tasks/{task_id}")
    async def get_task(self, task_id: str) -> Dict[str, Any]:
        response = _task_result(self.adapter.get_task_status_sync(task_id))
        response["task_id"] = response.pop("id", task_id)
        response["status_url"] = f"/v1/tasks/{task_id}"
        return response


app = PannsIngress.bind(PannsConsumer.bind(PannsEncoder.bind()))
