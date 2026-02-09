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
    """Lazy-initialize sherpa-onnx OnlineRecognizer (streaming transducer).

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
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(  # type: ignore[attr-defined]
            tokens=str(tokens),
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            num_threads=1,
            provider="cpu",
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            max_active_paths=4,
            lm="",
            lm_scale=0.0,
            lodr_fst="",
            lodr_scale=0.0,
            hotwords_file="",
            hotwords_score=1.5,
            modeling_unit="",
            bpe_vocab="",
            blank_penalty=0.0,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[sherpa] Failed to init recognizer: {exc}", flush=True)
        return None

    _RECOGNIZER = recognizer
    return recognizer


def transcribe_sherpa(audio: np.ndarray, sample_rate: int) -> str:
    """Transcribe a single utterance using sherpa-onnx OnlineRecognizer.

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

        # Add a small tail padding and mark end of input (mimic examples)
        tail = np.zeros(int(0.66 * sample_rate), dtype=np.float32)
        stream.accept_waveform(sample_rate, tail)  # type: ignore[attr-defined]
        stream.input_finished()  # type: ignore[attr-defined]

        # Decode until no streams are ready
        pending = [stream]
        while True:
            ready = [s for s in pending if recognizer.is_ready(s)]  # type: ignore[attr-defined]
            if not ready:
                break
            recognizer.decode_streams(ready)  # type: ignore[attr-defined]

        result = recognizer.get_result(stream)  # type: ignore[attr-defined]
        text = getattr(result, "text", "") or ""
        return text.strip()
    except Exception as exc:  # pragma: no cover
        print(f"[sherpa] transcribe error: {exc}", flush=True)
        return ""

