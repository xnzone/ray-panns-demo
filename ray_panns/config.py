"""Shared async-inference configuration.

The ingress deployment and every consumer replica must use exactly the same
queue and result backend. Keep this module free of Serve deployment imports so
it can be imported by both sides of the application graph.
"""

from __future__ import annotations

import os

from ray.serve.schema import CeleryAdapterConfig, TaskProcessorConfig

TASK_PREDICT_AUDIO = "panns.predict_audio"


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


BROKER_URL = os.getenv("PANNS_BROKER_URL", "redis://127.0.0.1:6379/0")
BACKEND_URL = os.getenv("PANNS_BACKEND_URL", BROKER_URL)
QUEUE_NAME = os.getenv("PANNS_QUEUE_NAME", "panns-inference")

# The built-in Celery adapter provides at-least-once delivery. Failed and
# unprocessable queues make exhausted retries observable instead of silently
# losing work.
PROCESSOR_CONFIG = TaskProcessorConfig(
    queue_name=QUEUE_NAME,
    adapter_config=CeleryAdapterConfig(
        broker_url=BROKER_URL,
        backend_url=BACKEND_URL,
    ),
    max_retries=_int_env("PANNS_MAX_RETRIES", 3),
    failed_task_queue_name=os.getenv("PANNS_FAILED_QUEUE", f"{QUEUE_NAME}.failed"),
    unprocessable_task_queue_name=os.getenv(
        "PANNS_UNPROCESSABLE_QUEUE", f"{QUEUE_NAME}.unprocessable"
    ),
)
