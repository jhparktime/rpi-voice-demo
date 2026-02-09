"""sherpa-onnx based STT backend for baseline mode."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import sherpa_onnx  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    _RECOGNIZER = None
else:
    _RECOGNIZER = None


def _init_recognizer() -> Optional[object]:
    """Lazy-initialize sherpa-onnx OfflineRecognizer.

    Expects model files under:

        sherpa_stt/
          tokens.txt
          encoder.onnx
          decoder.onnx
          joiner.onnx

    You can override the directory with SHERPA_STT_DIR env var.
    """
    global _RECOGNIZER
    if _RECOGNIZER is not None:
        return _RECOGNIZER

    try:
        sherpa_onnx  # type: ignore[name-defined]
    except Exception:
        return None

    root = Path(__file__).resolve().parent.parent
    model_dir = Path(os.environ.get("SHERPA_STT_DIR", root / "sherpa_stt"))
    tokens = model_dir / "tokens.txt"
    encoder = model_dir / "encoder.onnx"
    decoder = model_dir / "decoder.onnx"
    joiner = model_dir / "joiner.onnx"
    if not (tokens.exists() and encoder.exists() and decoder.exists() and joiner.exists()):
        print(f"[sherpa] STT models not found under {model_dir}; falling back.", flush=True)
        return None

    try:
        config = sherpa_onnx.OfflineRecognizerConfig(  # type: ignore[attr-defined]
            tokens=str(tokens),
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
        )
        recognizer = sherpa_onnx.OfflineRecognizer(config)  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover
        print(f"[sherpa] Failed to init recognizer: {exc}", flush=True)
        return None

    _RECOGNIZER = recognizer
    return recognizer


def transcribe_sherpa(audio: np.ndarray, sample_rate: int) -> str:
    """Transcribe a single utterance using sherpa-onnx OfflineRecognizer.

    Falls back to empty string if recognizer is not available.
    """
    recognizer = _init_recognizer()
    if recognizer is None:
        return ""

    # sherpa-onnx expects float32 mono waveform
    wav = audio.astype(np.float32)
    try:
        stream = recognizer.create_stream()  # type: ignore[attr-defined]
        stream.accept_waveform(sample_rate, wav)  # type: ignore[attr-defined]
        recognizer.decode_stream(stream)  # type: ignore[attr-defined]
        result = stream.result  # type: ignore[attr-defined]
        text = getattr(result, "text", "") or ""
        return text.strip()
    except Exception as exc:  # pragma: no cover
        print(f"[sherpa] transcribe error: {exc}", flush=True)
        return ""

