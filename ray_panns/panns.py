"""PANNs inference adapter.

The heavyweight dependency is imported lazily so the API, scheduler, and unit
tests can be imported without downloading a checkpoint.  The default backend
is the official ``panns-inference`` Cnn14 AudioSet model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

class PannsPredictor:
    """Load one process-local PANNs model and return top-k AudioSet scores."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        checkpoint = checkpoint_path or os.getenv("PANNS_CHECKPOINT")
        self.device = device or os.getenv("PANNS_DEVICE", "cuda")
        self.mock = os.getenv("PANNS_MOCK", "0").lower() in {"1", "true", "yes"}
        if self.mock:
            self.checkpoint_path = checkpoint or "mock://panns"
            self._labels = ["Speech", "Music", "Vehicle", "Animal", "Noise"]
            return
        if not checkpoint:
            raise ValueError("PANNS_CHECKPOINT or checkpoint_path is required")
        self.checkpoint_path = str(Path(checkpoint).expanduser())
        try:
            from panns_inference import AudioTagging, labels
            import librosa
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover - depends on deployment image
            raise RuntimeError(
                "Install panns-inference to use PANNsPredictor"
            ) from exc
        self._tagger = AudioTagging(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )
        self._labels = labels
        self._resample = librosa.resample
        self._read_audio = sf.read

    def predict(self, audio_path: str, top_k: int = 5) -> Dict[str, Any]:
        path = Path(audio_path).expanduser().resolve()
        if not path.is_file() and not self.mock:
            raise FileNotFoundError(f"audio file does not exist: {path}")
        if self.mock:
            return {
                "sample_rate": 16000,
                "original_sample_rate": 16000,
                "duration_seconds": 0.0,
                "embedding_dimensions": 2048,
                "mock": True,
                "tags": [
                    {"index": i, "label": label, "score": round(0.9 - i * 0.1, 3)}
                    for i, label in enumerate(self._labels[: max(1, min(top_k, len(self._labels)))])
                ],
            }
        waveform, sample_rate = self._read_audio(
            str(path), dtype="float32", always_2d=False
        )
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        original_sample_rate = sample_rate
        if sample_rate != 32000:
            waveform = self._resample(waveform, orig_sr=sample_rate, target_sr=32000)
            sample_rate = 32000
        clipwise_output, embedding = self._tagger.inference(waveform[None, :])
        import numpy as np

        scores = np.asarray(clipwise_output).reshape(-1)
        top_k = max(1, min(int(top_k), scores.size))
        indices = np.argsort(scores)[::-1][:top_k]
        return {
            "sample_rate": int(sample_rate),
            "duration_seconds": round(float(len(waveform) / sample_rate), 3),
            "embedding_dimensions": int(np.asarray(embedding).shape[-1]),
            "tags": [
                {"index": int(i), "label": self._labels[int(i)] if int(i) < len(self._labels) else f"audioset_class_{int(i)}", "score": float(scores[i])}
                for i in indices
            ],
            "original_sample_rate": int(original_sample_rate),
        }
