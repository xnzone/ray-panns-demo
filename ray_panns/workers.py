"""Serve deployments for the async consumer and the hot PANNs model."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from ray import serve
from ray.serve.config import AutoscalingConfig, AutoscalingPolicy
from ray.serve.task_consumer import task_consumer, task_handler

from .config import PROCESSOR_CONFIG, TASK_PREDICT_AUDIO
from .panns import PannsPredictor, load_audio


def _is_mock() -> bool:
    return os.getenv("PANNS_MOCK", "0").lower() in {"1", "true", "yes"}


def _consumer_deployment_options() -> Dict[str, Any]:
    """Add queue-depth autoscaling when supported by the installed Ray."""

    options: Dict[str, Any] = {
        "ray_actor_options": {
            "num_cpus": 1,
            "resources": {"panns_worker": 0.001},
        },
        "max_ongoing_requests": 1,
    }
    try:
        from importlib.util import find_spec

        policy_module = "ray.serve.async_inference_autoscaling_policy"
        if find_spec(policy_module) is None:
            return options
        policy = AutoscalingPolicy(
            policy_function=f"{policy_module}:AsyncInferenceAutoscalingPolicy",
            policy_kwargs={
                "broker_url": os.getenv(
                    "PANNS_BROKER_URL", "redis://127.0.0.1:6379/0"
                ),
                "queue_name": os.getenv("PANNS_QUEUE_NAME", "panns-inference"),
            },
        )
        options["autoscaling_config"] = AutoscalingConfig(
            min_replicas=1,
            max_replicas=4,
            target_ongoing_requests=1,
            policy=policy,
        )
    except (ImportError, TypeError, ValueError):
        # Ray versions before the async policy can still run the task consumer.
        pass
    return options


_CONSUMER_DEPLOYMENT_OPTIONS = _consumer_deployment_options()


@serve.deployment(
    num_replicas=1,
    ray_actor_options={
        "num_gpus": 0 if _is_mock() or os.getenv("PANNS_DEVICE", "cuda") == "cpu" else 1,
        "num_cpus": 1,
        "resources": {"panns_worker": 0.001},
    },
    max_ongoing_requests=1,
)
class PannsEncoder:
    """Standard Serve deployment that owns one warm PANNs model per replica."""

    def __init__(self):
        checkpoint = os.getenv("PANNS_CHECKPOINT", "")
        if not checkpoint and not _is_mock():
            raise RuntimeError("PANNS_CHECKPOINT must point to a downloaded PANNs checkpoint")
        self.predictor = PannsPredictor(checkpoint, os.getenv("PANNS_DEVICE", "cuda"))

    def healthz(self) -> Dict[str, str]:
        return {"status": "ready", "device": self.predictor.device}

    def predict(
        self, waveform: Optional[Any] = None, sample_rate: Optional[int] = None,
        top_k: int = 5, audio_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Mock mode accepts a path so local API tests do not need a soundfile or
        # a real checkpoint.
        if self.predictor.mock:
            return self.predictor.predict(audio_path or "mock://audio", top_k)
        if waveform is None or sample_rate is None:
            raise ValueError("waveform and sample_rate are required")
        return self.predictor.predict_waveform(waveform, int(sample_rate), top_k)


@serve.deployment(**_CONSUMER_DEPLOYMENT_OPTIONS)
@task_consumer(task_processor_config=PROCESSOR_CONFIG)
class PannsConsumer:
    """Queue consumer: CPU audio preparation plus RPC to the GPU encoder."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.mock = _is_mock()

    @task_handler(name=TASK_PREDICT_AUDIO)
    def predict_audio(self, audio_path: str, top_k: int = 5) -> Dict[str, Any]:
        if self.mock:
            return self.encoder.predict.remote(audio_path=audio_path, top_k=int(top_k)).result()

        waveform, sample_rate = load_audio(audio_path)
        # Celery acknowledges the message only after this result completes,
        # enabling safe redelivery of failed or lost work.
        return self.encoder.predict.remote(
            waveform=waveform, sample_rate=sample_rate, top_k=int(top_k)
        ).result()


# The task_consumer decorator wraps the class; give the Serve deployment a stable
# public name for RayService overrides and dashboards.
PannsConsumer = PannsConsumer.options(name="PannsConsumer")
