from __future__ import annotations

from typing import Any, Dict

import ray

from .panns import PannsPredictor


class _PannsWorkerImpl:
    """One serialized PANNs model per GPU actor."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.predictor = PannsPredictor(checkpoint_path, device)

    def warm_up(self) -> Dict[str, str]:
        return {"status": "ready", "device": self.predictor.device}

    def predict(self, audio_path: str, top_k: int = 5) -> Dict[str, Any]:
        return self.predictor.predict(audio_path, top_k)


@ray.remote(num_gpus=1, num_cpus=0, max_concurrency=1, max_restarts=1)
class PannsWorker(_PannsWorkerImpl):
    """GPU actor used when a CUDA device is available."""


@ray.remote(num_gpus=0, num_cpus=1, max_concurrency=1, max_restarts=1)
class PannsCpuWorker(_PannsWorkerImpl):
    """CPU variant used by OrbStack and other development environments."""
