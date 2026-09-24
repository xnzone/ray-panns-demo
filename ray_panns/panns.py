"""PANNs inference adapter and CPU-side audio preprocessing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def load_audio(audio_path: str) -> Tuple[np.ndarray, int]:
    """Read an audio file and convert stereo audio to a mono float32 waveform."""

    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("soundfile is required by the PANNs consumer") from exc

    waveform, sample_rate = sf.read(
        str(Path(audio_path).expanduser().resolve()),
        dtype="float32",
        always_2d=False,
    )
    waveform = np.asarray(waveform, dtype="float32")
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("audio file must contain a non-empty 1-D or 2-D waveform")
    return waveform, int(sample_rate)


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
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise RuntimeError("Install panns-inference and librosa to use PannsPredictor") from exc
        self._tagger = AudioTagging(checkpoint_path=self.checkpoint_path, device=self.device)
        self._labels = labels
        self._resample = librosa.resample

    def _mock_result(self, top_k: int) -> Dict[str, Any]:
        limit = max(1, min(int(top_k), len(self._labels)))
        return {
            "sample_rate": 16000,
            "original_sample_rate": 16000,
            "duration_seconds": 0.0,
            "embedding_dimensions": 2048,
            "mock": True,
            "tags": [
                {"index": i, "label": label, "score": round(0.9 - i * 0.1, 3)}
                for i, label in enumerate(self._labels[:limit])
            ],
        }

    def predict(self, audio_path: str, top_k: int = 5) -> Dict[str, Any]:
        path = Path(audio_path).expanduser().resolve()
        if self.mock:
            return self._mock_result(top_k)
        if not path.is_file():
            raise FileNotFoundError(f"audio file does not exist: {path}")
        waveform, sample_rate = load_audio(str(path))
        return self.predict_waveform(waveform, sample_rate, top_k=top_k)

    def predict_waveform(
        self, waveform: np.ndarray, sample_rate: int, top_k: int = 5,
        original_sample_rate: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run inference on a waveform prepared by the task consumer."""

        if self.mock:
            return self._mock_result(top_k)
        waveform = np.asarray(waveform, dtype="float32")
        original_sample_rate = int(original_sample_rate or sample_rate)
        if sample_rate != 32000:
            waveform = self._resample(waveform, orig_sr=sample_rate, target_sr=32000)
            sample_rate = 32000
        clipwise_output, embedding = self._tagger.inference(waveform[None, :])
        scores = np.asarray(clipwise_output).reshape(-1)
        top_k = max(1, min(int(top_k), scores.size))
        indices = np.argsort(scores)[::-1][:top_k]
        return {
            "sample_rate": int(sample_rate),
            "duration_seconds": round(float(len(waveform) / sample_rate), 3),
            "embedding_dimensions": int(np.asarray(embedding).shape[-1]),
            "tags": [
                {
                    "index": int(i),
                    "label": self._labels[int(i)] if int(i) < len(self._labels) else f"audioset_class_{int(i)}",
                    "score": float(scores[i]),
                }
                for i in indices
            ],
            "original_sample_rate": original_sample_rate,
        }
